# Winner Finder

Predict the winner of a wheel spin during live streams (from poíntauc.com) before it's determined.

## Description

Want to predict the result of a wheel spin before the winner is announced? Use this tool! Here's a preview of how it works:

![preview.gif](./preview.gif)

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

Run the program before the wheel spin starts:


```shell
streamlink --twitch-low-latency --stdout <channel link> best | python main.py winner

# or you can use a shorthand
./utils run <channel link>
```

## Lint

Run the following commands to lint your code:

```shell
pylint.
black .
isort .
```

## Test

Run the test suite with:

```shell
python -m unittest discover
```

## TODO list:

- Properly handle ellipsis cases (the wheel isn't always a perfect circle)
- Choose a more flexible solution for the config
- Improve linter setup