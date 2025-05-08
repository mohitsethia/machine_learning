import pandas as pd
import matplotlib.pyplot as plt

# Load both submissions
ours = pd.read_csv("submission.csv")
sample = pd.read_csv("sample_submission.csv")

# Merge on Id to align predictions
merged = ours.merge(sample, on="Id", suffixes=('_ours', '_sample'))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(merged["Id"], merged["SalePrice_ours"], label="Our Predictions", color="blue", alpha=0.7)
plt.plot(merged["Id"], merged["SalePrice_sample"], label="Sample Submission", color="orange", alpha=0.7)
plt.xlabel("Id")
plt.ylabel("SalePrice")
plt.title("Comparison of Predicted vs Sample Submission SalePrices")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
