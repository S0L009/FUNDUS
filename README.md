<div align="center">

# AutoFundus

![Fundus Image](flow.png)

</div>

## Hardware
# 🫷🫸

## Pipeline:

1. **Capturing High-Resolution Fundus Images:** Utilizing reinforcement learning for precise joystick movements to capture high-quality fundus images.
2. **Region of Interest Extraction:** Utilizing computer vision methods to isolate the ROI from the entire Fundus Image.
3. **Optic Disc Segmentation:** Isolating the optic disc from the cropped fundus images.
4. **Glaucoma Classification:** Differentiating between glaucoma suspects and healthy individuals
5. **Report Generation:** Creating a comprehensive report based on the classification results.
   
## Results:

| Prediction | Groundtruth |
|------------|-------------|
| ![Prediction Image](ppred_1.png) | ![Groundtruth Image](llabel_1.png) |

### Metrics
- Accuracy: 94.79
- Precision: 97.07
- Recall: 92.50
- F1 Score: 94.73
- Dice Score - 94.40
- IoU: 90.24

### Not Imp info:
The `data_prep.ipynb` was run on Windows, whereas the `.py` files were executed on Linux.
