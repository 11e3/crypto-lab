# Screenshot Instructions for README

## Quick Start

You've provided 10 excellent screenshots of the Crypto Lab. Here's how to prepare them for the README:

## Screenshots Mapping

Based on the images you shared, here's the mapping:

| Your Screenshot | Save As | Description |
|----------------|---------|-------------|
| Image 1 (Home page) | `docs/images/home_dashboard.png` | Home dashboard with features |
| Image 2 (Data Collection) | `docs/images/data_collection.png` | Ticker and interval selection |
| Image 3 (Backtest Settings - MinimalVBO) | `docs/images/backtest_settings.png` | Strategy configuration |
| Image 4 (Backtest Settings - VanillaVBO) | *(optional alternative)* | Different strategy params |
| Image 5 (Performance Overview) | `docs/images/backtest_results.png` | Performance metrics summary |
| Image 6 (Equity Curve) | `docs/images/equity_curve.png` | Interactive equity chart |
| Image 7 (Drawdown) | `docs/images/drawdown_chart.png` | Underwater curve |
| Image 8 (Yearly Returns) | `docs/images/yearly_returns.png` | Year-over-year bar chart |
| Image 9 (Statistics) | `docs/images/statistical_analysis.png` | Statistical significance |
| Image 10 (Optimization) | `docs/images/optimization.png` | Parameter optimization setup |
| Image 11 (Monte Carlo) | `docs/images/monte_carlo.png` | Monte Carlo simulation |

## Cropping Recommendations

### Image 1 - Home Dashboard
**Current**: Full width with sidebar
**Crop**:
- Left: Remove sidebar (crop from ~150px)
- Top: Keep title area
- Result: Focus on "Welcome to Crypto Lab Backtest" and features

### Image 2 - Data Collection
**Current**: Full data collection interface
**Crop**:
- Left: Remove sidebar
- Focus: Ticker selection + interval selection + summary
- Keep: All selected tickers (KRW-BTC, ETH, XRP, SOL, DOGE, TRX) and intervals

### Image 3 - Backtest Settings
**Current**: MinimalVBO strategy with "no configurable parameters" message
**Recommendation**: Use Image 4 (VanillaVBO) instead - shows actual parameters
**Crop**:
- Left: Remove sidebar
- Focus: Strategy selection, parameter sliders, trading costs, portfolio settings

### Image 5 - Performance Overview
**Current**: Full results with metrics
**Crop**:
- Top: Start from "Performance Summary"
- Include: All metrics grid (CAGR, Sharpe, MDD, etc.)
- Bottom: Stop after metrics, before "Trade History"

### Image 6 - Equity Curve
**Current**: Full equity curve tab
**Crop**:
- Top: Start from chart title "Portfolio Equity Curve"
- Focus: Just the chart area
- Include: Time range selector at bottom
- Remove: Tabs and surrounding UI

### Image 7 - Drawdown Chart
**Current**: Full underwater curve
**Crop**:
- Similar to equity curve
- Focus: Just the chart showing drawdown periods
- Include: MDD marker and label

### Image 8 - Yearly Returns
**Current**: Full yearly analysis with bar chart + metrics
**Crop**:
- Include: Bar chart + the 4 summary metrics below
- Focus: Clear view of positive/negative years

### Image 9 - Statistical Analysis
**Current**: Full statistics tab
**Crop**:
- Focus: "Statistical Significance Analysis" section
- Include: The table with Z-Score, P-Value, Skewness, Kurtosis
- Include: Result interpretation ("Not Significant" banner)

### Image 10 - Optimization
**Current**: Full optimization page
**Crop**:
- Remove sidebar
- Include: Strategy selection, parameter ranges, configuration summary
- Focus: Shows the "400 combinations" clearly

### Image 11 - Monte Carlo
**Current**: Full Monte Carlo simulation page
**Crop**:
- Remove sidebar
- Include: Strategy selection, simulation configuration
- Focus: Bootstrap resampling method, number of simulations slider

## Missing Screenshot

You'll need one more screenshot for the **Monthly Returns Heatmap**:

### To capture:
1. Go to Backtest > Results > Monthly Analysis tab
2. Scroll to show the monthly heatmap clearly
3. Make sure all 12 months are visible (now that we fixed the bug!)
4. Include the yearly totals at the bottom
5. Save as: `docs/images/monthly_heatmap.png`

## Automated Cropping (Recommended)

If you want automated cropping, you can use these tools:

### Option 1: Python + PIL
```bash
# Install Pillow if not already installed
uv pip install Pillow

# Use the crop script
uv run python scripts/crop_screenshots.py
```

### Option 2: ImageMagick
```bash
# Install ImageMagick first
# Then crop with:
magick input.png -crop 1200x800+150+50 output.png
```

### Option 3: Manual (Simplest)
1. Open each image in an image editor (Paint, GIMP, Photoshop)
2. Use the crop tool to remove:
   - Left sidebar (~150px)
   - Top header area (~50-100px)
   - Extra whitespace on right
3. Resize to ~1200px width if needed
4. Save as PNG with compression

## Quick Save Instructions

Since you already have the screenshots:

1. **Save your 11 images** to `c:\Users\moons\dev\crypto-lab\docs\images\`
2. **Rename them** according to the mapping table above
3. **Crop if needed** (optional - they look good as-is)
4. **Take the missing Monthly Heatmap** screenshot
5. **Commit to git**:
   ```bash
   git add docs/images/
   git commit -m "docs: Add screenshots for README visual documentation"
   ```

## Optimization

To reduce file sizes:

```bash
# Using Python
uv run python -c "
from PIL import Image
import os

images_dir = 'docs/images'
for filename in os.listdir(images_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(images_dir, filename)
        img = Image.open(img_path)
        img.save(img_path, optimize=True, quality=85)
        print(f'Optimized {filename}')
"
```

## Final Checklist

- [ ] All 11+ screenshots saved to `docs/images/`
- [ ] Files named according to convention
- [ ] Screenshots show realistic data (not empty states)
- [ ] Dark theme consistent across all screenshots
- [ ] File sizes reasonable (< 500KB each)
- [ ] Monthly heatmap screenshot captured (the missing one)
- [ ] All screenshots display correctly when viewing README.md
- [ ] Committed to git with descriptive message

## Viewing Results

After saving the screenshots, view the README:

```bash
# View in GitHub (push first)
git push origin main

# Or preview locally with a markdown viewer
# Or open in VS Code and preview with Ctrl+Shift+V
```

## Notes

- The screenshots you provided are excellent quality
- They show the system in use with real backtest data
- The dark theme looks professional
- Consider cropping to focus on specific features
- The current full-width screenshots are also acceptable if you prefer

## Alternative: Using Your Screenshots As-Is

If you want to use your screenshots without cropping:

1. Just save them with the correct filenames
2. They'll display full-width in the README
3. GitHub will automatically resize them
4. Still looks professional

The choice is yours - cropped (more focused) or full-width (more context).
