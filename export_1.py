import types
import argparse
import torch
import torch.nn.functional as F
import onnx
import onnxsim
from collections import OrderedDict # Import OrderedDict for robust input mapping

from modules.xfeat import XFeat

# --- Custom Modules/Functions from XFeat Library ---

class CustomInstanceNorm(torch.nn.Module):
    def __init__(self, epsilon=1e-5):
        super(CustomInstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), unbiased=False, keepdim=True)
        return (x - mean) / (std + self.epsilon)


def preprocess_tensor(self, x):
    # This is a bypass function from XFeat if not dynamic, assuming image dimensions
    # are multiples of 32. Not directly used for the matching-only export.
    return x, 1.0, 1.0

# This is an original XFeat method, kept for completeness of the XFeat class.
# It's not directly used by the 'mimic_lighterglue' export mode.
def match_xfeat_star(self, mkpts0, feats0, sc0, mkpts1, feats1, sc1):
    out1 = {
        "keypoints": mkpts0,
        "descriptors": feats0,
        "scales": sc0,
    }
    out2 = {
        "keypoints": mkpts1,
        "descriptors": feats1,
        "scales": sc1,
    }

    #Match batches of pairs
    idx0_b, idx1_b = self.batch_match(out1['descriptors'], out2['descriptors'] )

    #Refine coarse matches
    match_mkpts, batch_index = self.refine_matches(out1, out2, idx0_b, idx1_b, fine_conf = 0.25)

    return match_mkpts, batch_index


# --- ONNX Export Wrapper for XFeat mimicking LighterGlue Signature ---

def export_xfeat_mimic_lighterglue_onnx(
    self, mkpts0, feats0, image0_size, mkpts1, feats1, image1_size
):
    # Default min_cossim threshold for XFeat's batch_match.
    min_cossim_threshold = 0.75

    # Perform the core XFeat batch matching logic (mutual nearest neighbor)
    cossim = torch.bmm(feats0, feats1.permute(0, 2, 1)) # (B, N0, N1)

    cossim_max0, match12 = torch.max(cossim, dim=-1) # cossim_max0: (B, N0), match12: (B, N0)

    _, match21 = torch.max(cossim.permute(0, 2, 1), dim=-1) # match21: (B, N1)

    indices0 = (
        torch.arange(feats0.shape[1], device=feats0.device)
        .unsqueeze(0)
        .repeat(feats0.shape[0], 1)
    ) # (B, N0)

    mutual = (match21.gather(1, match12) == indices0) # (B, N0)

    if min_cossim_threshold > 0:
        good_scores_mask = (cossim_max0 > min_cossim_threshold) # (B, N0)
        mutual = mutual & good_scores_mask # Combine masks

    mutual_indices_b_flat = mutual.nonzero() # (total_num_matches_across_batches, 2) [batch_idx, kp_idx_0]

    matched_kp1_idx_flat = match12[mutual] # (total_num_matches_across_batches)

    idx0_filtered = mutual_indices_b_flat[:, 1] # Keypoint index from image 0
    idx1_filtered = matched_kp1_idx_flat        # Keypoint index from image 1

    # --- CRITICAL CHANGE: DO NOT UNSQUEEZE FOR BATCH DIMENSION FOR OUTPUTS ---
    # Outputs are now [num_matches, 2] and [num_matches]
    matches_output = torch.stack((idx0_filtered, idx1_filtered), dim=-1).to(
        torch.int64
    ) # Shape: [num_matches_total, 2]

    scores_output = cossim_max0[mutual] # Shape: [num_matches_total]


    # --- Dummy dependency for unused inputs (mkpts0, mkpts1, image0_size, image1_size) ---
    # These inputs are required by the LighterGlue signature but not by XFeat's MNN logic.
    # We must ensure they contribute to the graph to prevent ONNX from pruning them.
    # This involves a numerically zero-effect operation.

    # 1. Sum elements of 1D image_size inputs (now shape [2])
    # The sum will be a scalar.
    dummy_sum_image0_size = torch.sum(image0_size.float()) 
    dummy_sum_image1_size = torch.sum(image1_size.float()) 

    # 2. Sum elements of keypoint inputs (mkptsX still have batch dimension [B, N, 2])
    # Summing across N and 2 dimensions leaves a [B] tensor.
    dummy_sum_mkpts0 = torch.sum(mkpts0.float(), dim=(1,2)) 
    dummy_sum_mkpts1 = torch.sum(mkpts1.float(), dim=(1,2)) 

    # 3. Combine all dummy sums into a single scalar that's added to the scores output.
    # We need to sum dummy_sum_mkpts0/1 (which are [B]) into a scalar, and add the scalar image_size sums.
    # This dummy_dependency_scalar will be a single number.
    dummy_dependency_scalar = torch.sum(dummy_sum_mkpts0) + torch.sum(dummy_sum_mkpts1) + \
                              dummy_sum_image0_size + dummy_sum_image1_size

    # 4. Add this scalar (multiplied by zero) to the scores output.
    # This creates a computational dependency in the ONNX graph but does not change the actual scores.
    # scores_output is 1D: [num_matches_total]. Adding a scalar works via broadcasting.
    scores_output = scores_output + dummy_dependency_scalar * 0.0

    return matches_output, scores_output


# --- Argument Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(description="Export XFeat/Matching model to ONNX.")
    # General XFeat export options
    parser.add_argument("--xfeat_only_model", action="store_true", help="Export only the XFeat model (detection backbone).")
    parser.add_argument("--xfeat_only_model_detectAndCompute", action="store_true", help="Export the XFeat detectAndCompute model (sparse features).")
    parser.add_argument("--xfeat_only_model_dualscale", action="store_true", help="Export only the XFeat dualscale model (dense features).")
    parser.add_argument("--xfeat_only_matching", action="store_true", help="Export only the XFeat star matching (coarse+refinement).")
    parser.add_argument("--xfeat_only_lighterglue", action="store_true", help="Export only the XFeat Lighterglue addon matching model (original LightGlue Python wrapper).")
    parser.add_argument("--split_instance_norm", action="store_true", help="Whether to split InstanceNorm2d into '(x - mean) / (std + epsilon)', due to some inference libraries not supporting InstanceNorm, such as OpenVINO.")

    # Model configuration
    parser.add_argument("--height", type=int, default=960, help="Input image height (for detection models).")
    parser.add_argument("--width", type=int, default=1920, help="Input image width (for detection models).")
    parser.add_argument("--top_k", type=int, default=1024, help="Keep best k features.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes for batch and keypoint count.")

    # ONNX Export specific options
    parser.add_argument("--export_path", type=str, default="./match.onnx", help="Path to export ONNX model.")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version.")

    # THIS IS THE SPECIFIC EXPORT MODE FOR YOUR USE CASE
    parser.add_argument(
        "--xfeat_mimic_lighterglue_matching",
        action="store_true",
        help="Export XFeat batch matcher with LighterGlue's ONNX input/output signature (image_size=[2], outputs no batch dim).",
    )

    return parser.parse_args()


# --- Main Execution Block ---

if __name__ == "__main__":
    args = parse_args()

    # Determine dummy batch size and keypoint count based on dynamic flag
    if args.dynamic:
        batch_size = 1 # We use 1 for dummy export but mark as dynamic where possible
        num_kpts_dummy_0 = 1000 
        num_kpts_dummy_1 = 1000
    else:
        # Fixed sizes if not dynamic; ensure multiples of 32 for detection models
        assert args.height % 32 == 0 and args.width % 32 == 0, "Height and width must be multiples of 32."
        batch_size = 1
        num_kpts_dummy_0 = args.top_k
        num_kpts_dummy_1 = args.top_k


    if args.top_k > 4800:
        print("Warning: The current maximum supported value for TopK in TensorRT is 3840, which coincidentally equals 4800 * 0.8. Please ignore this warning if TensorRT will not be used in the future.")

    # Initialize XFeat model
    xfeat = XFeat()
    xfeat.top_k = args.top_k

    if args.split_instance_norm:
        xfeat.net.norm = CustomInstanceNorm()

    xfeat = xfeat.cpu().eval() # Ensure model is on CPU and in evaluation mode
    xfeat.dev = "cpu"
    args.xfeat_mimic_lighterglue_matching = True
    # Bypass preprocess_tensor if not dynamic and not a full pipeline model (e.g., just matching)
    if not args.dynamic and not (args.xfeat_only_matching or args.xfeat_only_lighterglue or args.xfeat_mimic_lighterglue_matching):
         xfeat.preprocess_tensor = types.MethodType(preprocess_tensor, xfeat)


    # --- Export Logic based on Command Line Arguments ---

    # Export the XFeat matcher mimicking LighterGlue's signature
    if args.xfeat_mimic_lighterglue_matching:
        print("Exporting XFeat batch matcher with LighterGlue ONNX signature (forcing all inputs, no batch dim on outputs).")
        xfeat.forward = types.MethodType(export_xfeat_mimic_lighterglue_onnx, xfeat)

        # Create dummy inputs that match the desired ONNX graph structure:
        # mkptsX: [batch, num_keypoints, 2]
        # featsX: [batch, num_keypoints, 64]
        # imageX_size: [2] (rank 1)
        mkpts0_dummy = torch.randn(batch_size, num_kpts_dummy_0, 2, dtype=torch.float32, device='cpu')
        feats0_dummy = torch.randn(batch_size, num_kpts_dummy_0, 64, dtype=torch.float32, device='cpu')
        image0_size_dummy = torch.randn(2, dtype=torch.float32, device='cpu') # Shape: [2]

        mkpts1_dummy = torch.randn(batch_size, num_kpts_dummy_1, 2, dtype=torch.float32, device='cpu')
        feats1_dummy = torch.randn(batch_size, num_kpts_dummy_1, 64, dtype=torch.float32, device='cpu')
        image1_size_dummy = torch.randn(2, dtype=torch.float32, device='cpu') # Shape: [2]

        # Use OrderedDict to explicitly define input names and their corresponding dummy tensors.
        # This is more robust against subtle reordering issues.
        inputs = OrderedDict([
            ("mkpts0", mkpts0_dummy),
            ("feats0", feats0_dummy),
            ("image0_size", image0_size_dummy),
            ("mkpts1", mkpts1_dummy),
            ("feats1", feats1_dummy),
            ("image1_size", image1_size_dummy),
        ])

        # Define dynamic axes for inputs and outputs.
        # Outputs now have 'num_matches' as their first (and only) dynamic dimension.
        dynamic_axes = {
            "mkpts0": {0: "batch", 1: "num_keypoints_0"},
            "feats0": {0: "batch", 1: "num_keypoints_0"},
            # image_size inputs are 1D ([2]), so no batch dim to mark dynamic
            "mkpts1": {0: "batch", 1: "num_keypoints_1"},
            "feats1": {0: "batch", 1: "num_keypoints_1"},
            "matches": {0: "num_matches"}, # Output matches: [num_matches, 2]
            "scores": {0: "num_matches"},  # Output scores: [num_matches]
        }

        # Define the output ONNX path specifically for this mode
        output_onnx_path = "./match.onnx" 
        print(f"Saving ONNX model to: {output_onnx_path}")

        # Perform the ONNX export
        torch.onnx.export(
            xfeat,
            tuple(inputs.values()), # Pass tensors from the OrderedDict
            output_onnx_path,
            verbose=False,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=list(inputs.keys()), # Pass names from the OrderedDict
            output_names=["matches", "scores"],
            dynamic_axes=dynamic_axes if args.dynamic else None,
        )

    # --- Other Export Modes (Original XFeat Options) ---
    # These blocks are mutually exclusive with the mimic mode due to 'elif'.

    elif args.xfeat_only_model:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
        torch.onnx.export(
            xfeat.net,
            (torch.randn(batch_size, 3, args.height, args.width, dtype=torch.float32, device='cpu')),
            args.export_path, verbose=False, opset_version=args.opset, do_constant_folding=True,
            input_names=["images"], output_names=["feats", "keypoints", "heatmaps"],
            dynamic_axes=dynamic_axes if args.dynamic else None,)

    elif args.xfeat_only_model_detectAndCompute:
        print("Warning: Exporting the detectAndCompute ONNX model only supports a batch size of 1.")
        xfeat.forward = xfeat.detectAndCompute
        x1 = torch.randn(batch_size, 3, args.height, args.width, dtype=torch.float32, device='cpu')
        dynamic_axes = {"images": {2: "height", 3: "width"}}
        torch.onnx.export(
            xfeat,
            (x1, args.top_k),
            args.export_path, verbose=False, opset_version=args.opset, do_constant_folding=True,
            input_names=["images", "top_k"], output_names=["keypoints", "scores", "descriptors"],
            dynamic_axes=dynamic_axes if args.dynamic else None,)

    elif args.xfeat_only_model_dualscale:
        xfeat.forward = xfeat.detectAndComputeDense
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
        torch.onnx.export(
            xfeat,
            (torch.randn(batch_size, 3, args.height, args.width, dtype=torch.float32, device='cpu'), args.top_k),
            args.export_path, verbose=False, opset_version=args.opset, do_constant_folding=True,
            input_names=["images"], output_names=["mkpts", "feats", "sc"],
            dynamic_axes=dynamic_axes if args.dynamic else None,)

    elif args.xfeat_only_matching: # This is the original match_xfeat_star (produces refined coordinates)
        xfeat.forward = types.MethodType(match_xfeat_star, xfeat)
        mkpts0 = torch.randn(batch_size, num_kpts_dummy_0, 2, dtype=torch.float32, device='cpu')
        mkpts1 = torch.randn(batch_size, num_kpts_dummy_1, 2, dtype=torch.float32, device='cpu')
        feats0 = torch.randn(batch_size, num_kpts_dummy_0, 64, dtype=torch.float32, device='cpu')
        feats1 = torch.randn(batch_size, num_kpts_dummy_1, 64, dtype=torch.float32, device='cpu')
        sc0 = torch.randn(batch_size, num_kpts_dummy_0, dtype=torch.float32, device='cpu')
        sc1 = torch.randn(batch_size, num_kpts_dummy_1, dtype=torch.float32, device='cpu')
        dynamic_axes = {
            "mkpts0": {0: "batch", 1: "num_keypoints_0"}, "feats0": {0: "batch", 1: "num_keypoints_0", 2: "descriptor_size"}, "sc0": {0: "batch", 1: "num_keypoints_0"},
            "mkpts1": {0: "batch", 1: "num_keypoints_1"}, "feats1": {0: "batch", 1: "num_keypoints_1", 2: "descriptor_size"}, "sc1": {0: "batch", 1: "num_keypoints_1"},
        }
        torch.onnx.export(
            xfeat,
            (mkpts0, feats0, sc0, mkpts1, feats1, sc1),
            args.export_path, verbose=False, opset_version=args.opset, do_constant_folding=True,
            input_names=["mkpts0", "feats0", "sc0", "mkpts1", "feats1", "sc1"],
            output_names=["matches", "batch_indexes"],
            dynamic_axes=dynamic_axes if args.dynamic else None,)

    elif args.xfeat_only_lighterglue: # Original LighterGlue ONNX export code (from modules.lighterglueonnx)
        if args.opset < 14:
            print(f"Lighterglue requires at least opset 14, bumping from {args.opset}")
            args.opset = 14
        # Original LighterGlue also expects [1, N, 2] mkpts and [2] image_size inputs
        mkpts0 = torch.randn(1, args.top_k, 2, dtype=torch.float32, device='cpu')
        mkpts1 = torch.randn(1, args.top_k, 2, dtype=torch.float32, device='cpu')
        feats0 = torch.randn(1, args.top_k, 64, dtype=torch.float32, device='cpu')
        feats1 = torch.randn(1, args.top_k, 64, dtype=torch.float32, device='cpu')
        image0_size = torch.randn(2, dtype=torch.float32, device='cpu') # This is the [2] shape
        image1_size = torch.randn(2, dtype=torch.float32, device='cpu') # This is the [2] shape
        dynamic_axes = {
            "mkpts0": {1: "num_keypoints_0"}, "feats0": {1: "num_keypoints_0"},
            "mkpts1": {1: "num_keypoints_1"}, "feats1": {1: "num_keypoints_1"},
            "matches": {0: "num_matches"}, # LighterGlue outputs are also typically [num_matches, 2]
            "scores": {0: "num_matches"},  # LighterGlue outputs are also typically [num_matches]
        }
        from modules.lighterglueonnx import LighterGlueONNX
        lighterglue = LighterGlueONNX()
        lighterglue = lighterglue.eval().cpu()
        torch.onnx.export(
            lighterglue,
            (mkpts0, feats0, image0_size, mkpts1, feats1, image1_size),
            args.export_path, verbose=False, opset_version=args.opset, do_constant_folding=True,
            input_names=["mkpts0", "feats0", "image0_size", "mkpts1", "feats1", "image1_size"],
            output_names=["matches", "scores"],
            dynamic_axes=dynamic_axes if args.dynamic else None,)

    # else: # Default behavior if no specific flag is given: Export full XFeat pipeline
    #     x1 = torch.randn(batch_size, 3, args.height, args.width, dtype=torch.float32, device='cpu')
    #     x2 = torch.randn(batch_size, 3, args.height, args.width, dtype=torch.float32, device='cpu')
    #     xfeat.forward = xfeat.match_xfeat_star
    #     dynamic_axes = {"images0": {0: "batch", 2: "height", 3: "width"}, "images1": {0: "batch", 2: "height", 3: "width"}}
    #     torch.onnx.export(
    #         xfeat,
    #         (x1, x2),
    #         args.export_path, verbose=False, opset_version=args.opset, do_constant_folding=True,
    #         input_names=["images0", "images1"], output_names=["matches", "batch_indexes"],
    #         dynamic_axes=dynamic_axes if args.dynamic else None,)


    # --- Post-Export Verification and Simplification ---

    # Determine which model path was used in the export block above
    final_output_path = args.export_path
    if args.xfeat_mimic_lighterglue_matching:
        final_output_path = "./match.onnx"
    # Note: If other specific modes override args.export_path, they should update final_output_path too.

    model_onnx = onnx.load(final_output_path)  # Load the exported ONNX model
    onnx.checker.check_model(model_onnx)  # Check ONNX model validity
    
    print(f"\n--- ONNX Model Details (before onnxsim.simplify) for {final_output_path} ---")
    print(model_onnx) # Print the ONNX graph structure to confirm inputs/outputs
    print(f"--- End of ONNX Model Details (before simplify) ---")

    # Simplify the ONNX model using onnxsim
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "ONNX simplification failed!"
    onnx.save(model_onnx, final_output_path) # Save the simplified model

    print(f"\n--- ONNX Model Details (AFTER onnxsim.simplify) for {final_output_path} ---")
    print(model_onnx) # Print the simplified ONNX graph structure
    print(f"--- End of ONNX Model Details (after simplify) ---")

    print(f"\nModel exported and simplified to {final_output_path}")