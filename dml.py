import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

# Create figure for DML flowchart
fig, ax = plt.subplots(figsize=(12, 8))

# Step 1: Data Input (Stacked Rectangles for Hard Drive/Dataset)
ax.add_patch(Rectangle((0.3, 0.85), 0.4, 0.1, fill=True, color='#001F3F', alpha=1.0))
ax.add_patch(Rectangle((0.32, 0.8), 0.36, 0.1, fill=True, color='#001F3F', alpha=0.8))
ax.add_patch(Rectangle((0.34, 0.75), 0.32, 0.1, fill=True, color='#001F3F', alpha=0.6))
plt.text(0.5, 0.9, 'Step 1: Data Input\nLarge dataset: Treatment (e.g., ad campaign),\nOutcome (e.g., iCLV), Confounders (age, income)',
         ha='center', va='center', fontsize=10, color='white')

# Step 2: Predicting with ML (Rectangle for simplicity)
ax.add_patch(Rectangle((0.3, 0.55), 0.4, 0.15, fill=True, color='#28A745'))
plt.text(0.5, 0.625, 'Step 2: Predicting with ML\nTwo models predict:\n- Treatment (ad exposure)\n- Outcome (iCLV)',
         ha='center', va='center', fontsize=10, color='white')

# Step 3: Cross-Fitting (Circle for iterative process)
ax.add_patch(Circle((0.5, 0.35), 0.15, fill=True, color='#F0F0F0'))
plt.text(0.5, 0.35, 'Step 3: Cross-Fitting\nSplit data into folds, predict\nusing other folds to avoid overfitting',
         ha='center', va='center', fontsize=10)

# Step 4: Combining Estimates (Rectangle for final output)
ax.add_patch(Rectangle((0.3, 0.05), 0.4, 0.15, fill=True, color='#FFA500'))
plt.text(0.5, 0.125, 'Step 4: Combining Estimates\nMerge predictions with actual results\nto estimate unbiased causal effect',
         ha='center', va='center', fontsize=10, color='white')

# Arrows to show flow
# Step 1 to Step 2
arrow1 = FancyArrowPatch((0.5, 0.75), (0.5, 0.7), connectionstyle="arc3,rad=0", arrowstyle='->', color='black', linewidth=1.5)
ax.add_patch(arrow1)

# Step 2 to Step 3 (Three arrows for cross-fitting folds)
arrow2a = FancyArrowPatch((0.5, 0.55), (0.45, 0.5), connectionstyle="arc3,rad=-0.2", arrowstyle='->', color='black', linewidth=1)
arrow2b = FancyArrowPatch((0.5, 0.55), (0.5, 0.5), connectionstyle="arc3,rad=0", arrowstyle='->', color='black', linewidth=1)
arrow2c = FancyArrowPatch((0.5, 0.55), (0.55, 0.5), connectionstyle="arc3,rad=0.2", arrowstyle='->', color='black', linewidth=1)
ax.add_patch(arrow2a)
ax.add_patch(arrow2b)
ax.add_patch(arrow2c)

# Step 3 to Step 4 (Three arrows converging)
arrow3a = FancyArrowPatch((0.45, 0.35), (0.4, 0.2), connectionstyle="arc3,rad=-0.2", arrowstyle='->', color='black', linewidth=1)
arrow3b = FancyArrowPatch((0.5, 0.35), (0.5, 0.2), connectionstyle="arc3,rad=0", arrowstyle='->', color='black', linewidth=1)
arrow3c = FancyArrowPatch((0.55, 0.35), (0.6, 0.2), connectionstyle="arc3,rad=0.2", arrowstyle='->', color='black', linewidth=1)
ax.add_patch(arrow3a)
ax.add_patch(arrow3b)
ax.add_patch(arrow3c)

# Clean up and title
plt.title('How Double Machine Learning (DML) Works – Step by Step for Non-Technical Readers', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig('dml_flow_enhanced.png')  # Save for newsletter
plt.show()