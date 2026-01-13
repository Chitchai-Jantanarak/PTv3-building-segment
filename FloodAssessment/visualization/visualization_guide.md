# Point Cloud Visualization Guide (CloudCompare)

This guide explains how to inspect the output of the Seg-B model (Segmentation & Inpainting) using CloudCompare.

## 1. Loading the Data

1.  Open CloudCompare.
2.  File -> Open -> Select your generated `.las` file (e.g., `visualization_output.las`).
3.  Click "Apply" (you can accept the default Global Shift suggestions).

## 2. Visualizing Segmentation Classes

The tool adds a Scalar Field called `PredClass`.

1.  Select the cloud in the DB Tree on the left.
2.  In the Properties window, find "Scalar Fields".
3.  Change the "Active" field to **"PredClass"**.
4.  You will see points colored by class:
    -   **0**: Terrain (Usually Blue/Cold color)
    -   **1**: Building (Usually Red/Hot color)

## 3. Visualizing Inpainting Magnitude

The tool adds a Scalar Field called `InpaintMag` (Magnitude of the inpainting shift).

1.  Change the "Active" scalar field to **"InpaintMag"**.
2.  Adjust the Color Scale (View -> Color Scale Manager) if needed to see subtle differences.
3.  This shows how *much* the model decided to move/change a point (or the confidence/value of the property).

## 4. Filtering (Isolating Buildings)

To see *only* what the inpainting did to the buildings:

1.  Select the cloud.
2.  Go to **Edit -> Scalar Fields -> Filter by Value**.
3.  Select **"PredClass"** as the field.
4.  Set Range: **min=1, max=1**.
5.  Click **Export**.
6.  A new cloud "extracted" will appear in the tree. Hide the original.
7.  Now, on this new cloud, set the Active Scalar Field to **"InpaintMag"**.
    -   Now you are visualizing the inpainting intensity *only on the buildings*.

## 5. Comparing Against Ground Truth (Cloud-to-Cloud Distance)

If you have a ground truth LAS file (e.g., the original undamaged scan):

1.  Open the Ground Truth LAS file.
2.  Select BOTH the Prediction Cloud and the Ground Truth Cloud (Ctrl+Click).
3.  Go to **Tools -> Distances -> Cloud/Cloud Dist**.
4.  Click **Compute**.
5.  This generates a heatmap showing exactly where the geometry differs.
