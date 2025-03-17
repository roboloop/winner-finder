# Winner Finder

Predict the winner of a wheel spin during live streams (from poíntauc.com) before it's determined.

## Description

Want to predict the result of a wheel spin before the winner is announced? Use this tool! Here's a preview of how it works:

![preview.gif](./preview.gif)

## How it works

The solution is entirely built on collected heuristics and on standard human behavioral patterns. The implementation relies on screen parsing via OpenCV, text recognition via Tesseract OCR engine and basic math computations.

The program workflow:

1. Detecting the initial spin frame

   - The program processes frames sequentially until it finds one that meets the following criteria:
   - The current frame contains a wheel (identified as a large circle) and has the text "Winner" above it.
   - The next frame shows the wheel in motion (determined by comparing the angular difference between frames).
   
2. Determining the total spin duration

   - The simplest approach is detecting a user-entered duration on the screen (by parsing an image and extracting a numerical value)
   - If the duration box is unrecognized, hidden, or spin duration is randomized, a mathematical approach is applied. The program estimates the duration based on observed wheel movement. The wheel spin follows a custom easing function implemented with [GSAP](https://gsap.com/docs/v3/Eases/CustomEase/), which optimizes Bézier curves by generating interpolated `x` and `y` values for improved calculation accuracy. However, this implementation introduces slight deviations near the interpolated points. These deviations help eliminate unsuitable duration candidates, typically narrowing the range to a 1-4 second window. Post-processing further refines the estimate. This approximation is sufficient for our needs at this stage
   - The fallback strategy: if all else fails, the program prompts the user for input

3. Calculating the destination angle

    Picking a frame from the spinning sequence:

   - $x_{i} = \frac{d_{i}}{d}$, where `dᵢ` is the elapsed time, and `d` is the total spin duration
   - $y_{i} = \frac{a_{i}}{a}$, where `aᵢ` is the elapsed angular displacement (including full rotations) from initial state, and `a` is the target angle
   - $y_{i} = bezier(x_{i})$, where the `bezier` function maps the elapsed time to the corresponding elapsed angle, both normalized to a scale from 0 to 1
   - The target angle is computed as: $a = \frac{a_{i}}{bezier(\frac{d_{i}}{d})}$

4. Collecting lot names and their positions on the wheel

   - The initial frame is used to determine sector boundaries (collected in format start_angle and end_angle)
   - The initial wheel spin is analyzed to extract sector names (obtained from the text displayed above the wheel)

5. Refining calculations for greater accuracy

   - Avoiding early-stage angle measurements due to high error margins
   - Smoothing data by filtering out spikes, duplicates, and inconsistencies
   - Discarding invalid duration candidates based on discrepancies between computed and expected target angles
   - Converting the elliptical wheel projection to a circular model for more precise measurements (not yet implemented)
   - Extending the spin analysis window to improve text recognition of lot names
   - Implementing a voting system to enhance accuracy in determining the winning sector

## Installation

1. Install the project:

    ```shell
    # clone the repository
    git clone https://github.com/roboloop/winner-finder

    # create a virtual environment
    python -m venv myenv

    # activate the virtual environment
    source ./.venv/bin/activate

    # install dependencies
    pip install -r requirements.txt
    ```

2. Install [tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html), which is required for text recognition in images. Make sure to install the necessary language packs from the [tessdata](https://github.com/tesseract-ocr/tessdata) repository.

3. Install [streamlink](https://github.com/streamlink/streamlink), which allows you to grab a video stream from any platforms (Twitch, YouTube, Kick, etc...).

   > ℹ️ [yt_dlp](https://github.com/yt-dlp/yt-dlp) can be used as alternative.

## Run

Run the program before the wheel starts to spin:


```shell
streamlink --twitch-low-latency --stdout <channel link> best | python main.py winner

# or you can use a shorthand
./utils run <channel link>

# or you can analyze your video snippet
cat vod.ts | python main.py winner
```

## Lint

Run the following commands to lint your code:

```shell
black .
```

## Test

Run the test suite with:

```shell
python -m unittest discover -s tests -t .
```

## TODO list:

- Properly handle ellipsis cases (the wheel isn't always a perfect circle)
- Choose a more flexible solution for the config
- Fix GitHub Actions
- Improve linter setup