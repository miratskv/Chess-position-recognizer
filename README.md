## Short description:
A simple application designed for recognizing "online" chess positions (i.e., corresponding FEN notation) from a screenshot using a small neural network and predicting next best moves, utilizing Stockfish engine. The package includes: An application script, a script for generating training data and a script for training of the mentioned network.

### Disclaimer:
The application primarily serves for analyzing live broadcasts and facilitating an easy import of games from videos into an analyzer etc. Secondary purpose is purely educational, as the ability to generate custom data enables easy comparison of the effectiveness of various neural network models, the influence of data augmentation, and similar aspects. Any misuse for cheating in one's own games is immoral and honestly quite stupid.


## Motivation:
During the last few years, the popularity of chess has been rapidly increasing. This is accompanied by a growth in online content, whether it's educational videos, game recordings, or live broadcasts of matches at the highest level.

It is evident that modern chess engines far surpass even the most accomplished human players in terms of their performance. These engines have become indispensable tools for not only game analysis but also for evaluating the current state of a chess position, often quantified using metrics like the "Centipawn metric."

However, particularly in the realm of game streaming, a common scenario unfolds where viewers lack access to real-time position evaluation, let alone the best possible subsequent moves. During these live games, viewers often find themselves in a position where they wish they could see which player has the upper hand, understand the strategic moves necessary to maintain a lead, or even verify the feasibility of their own ideas within a specific game position. Standard chess websites naturally include tools for game analysis; however, setting up a position based on an image is impractical and time-consuming for an average person.

The purpose of this application is thus quite evident: Take a screenshot -> display the best following moves along with evaluations -> for easy position export, also generate the standard FEN notation.



## Requirements:

- **Operating System:** Tested only on a Linux machine (Ubuntu 23.04)
- **Chess engine (Optional):** Stockfish + python stockfish api (see [Stockfish on PyPI](https://pypi.org/project/stockfish/)). If the engine is not installed, app will show only the predicted FEN without the evaluation and best lines.
- **GUI framework:** Tkinter (see [How to Install Tkinter on Linux](https://www.geeksforgeeks.org/how-to-install-tkinter-on-linux/))
- **Python >= 3.11:**
  - tensorflow >= 2.12.0
  - numpy
  - pillow
  - opencv-python
  - stockfish

## Manual
After successful installation (you can use, for example, pip install command), the application itself is launched through the command line using the command "chessrec_app". Optional parameters:

  -  &#45;&#45;master_H": Dimension of the app window (default 540)
  -  &#45;&#45;master_W: Dimension of the app window (default 360)
  -  &#45;&#45;stockfish_elo: Level of the stockfish engine (default 3000)
  -  &#45;&#45;stockfish_depth: Depth of the tree search of the engine (default 16)
  -  &#45;&#45;stockfish_hash: Memory usage (default 2048)
  -  &#45;&#45;stockfish_threads: Number of parallel threads the engine uses (default 4)

Note that if the Stockfish engine is not installed globally, you have to set the correct path to the bin files (see the link in the previous sect.) in the constants.py file.

1. Press the "Select area" button to open a new transparent window. Resize and position this window over the area from which you want to take a screenshot of the chessboard. Edges do not have to be aligned properly, however too big offsets can downgrade the precision. Note that you can reselect and adjust the selected area by pressing the button again.

2. Use the auxiliary buttons to select which player is to move and the chessboard perspective. Additionally, select which castling options are still permissible.

3. Upon clicking the "Evaluate" button, the left portion will display a screenshot of the selected area, while the right side will show the neural network's interpretation of the given position for verification. In the text section, you'll find several best moves along with their centipawn evaluations and line continuation. Below that, you will also see the FEN notation.

Note: The FEN notation, of course, isn't complete, as it's impossible to determine the current move count, repetition history, or en passant option solely from an image.

![Example Image](example.png)
*Example: GM Daniel Naroditsky ([YouTube Link](https://www.youtube.com/watch?v=bFLEuc7G7YA)) explaining his thought process. On the right side of the image, we see the application: Buttons are set for the perspective of the black player, the black player is to move, and all castling options are permissible. The left board is the screenshot, the right one is the reconstruction based on the neural network recognition (hence the default "white" orientation). As you can see, the position was reconstructed correctly in this case. The position's evaluation is displayed in the text box, and each use of the "Evaluate" button is separated by a horizontal line. We can observe that GM Naroditsky briefly considers the third-best move, knight from D5 to B6. (Spoiler) However, he ultimately decides to play e7 to e6, which, as we can see, is by far the top engine move.*


## Code description

### Main directory
- **constants.py** file with global constants and parameters, such as stockfish path, number of files/ranks, pieces notations etc.
- **fen_transcode.py:** Utility functions for translating between categorical encoding, FEN notation and image generation from the mentioned
- **data_generator.py:** ChessBoardGenerator class. For a given set of training assets, such as backgrounds, different board styles and different pieces styles, randomly generates images of chess boards with the corresponding label. Public method tfGenerator() returns a tensorflow dataset generator. Example of required data structure: /cmds/generator_assets_example

### assets
Default assets for reconstruting screenshot image.

### app
- **app_buttons.py** implementation of the helper app's buttons, such as player to move, board orientation etc.
- **app_main.py:** App class source code

### engine interface
Since the python stockfish api only allows to return the best moves, some minor modifications are needed to be able to see whole lines. interface.py then just implement basic functions neede for the app.

### models
Include implementation of custom metrics (such as BoardAccuracy), model base class and the neural network architecture itslef.

### cmds
Command line scripts installed together with the package

## Chessboard Data Generation Process

In this document, we outline the process of generating annotated training data for chess models using appropriate images of chessboards and pieces. The main source of motifs used for this purpose is available at [lichess/lila GitHub repository](https://github.com/lichess-org/lila). The data generator source code can be found in the file `data_generator.py`.


- Position Generation: The arrangement of chess pieces on the board is established through a nearly random selection process. The total number of pieces is chosen randomly, and the relative frequency of each piece type is determined by its typical starting representation in the game. Note that we completly ignore whether the resulting position is realistic/legal. This approach serves as a useful data regularization technique, reducing bias towards common positions and encouraging the network to learn from diverse scenarios and/or work with same efficiency even for modified rules like Fischer's random chess etc.

- Background Considerations: The practical usability of the generated data is a key consideration. Users might not always provide the model with perfectly cropped square images. To accommodate this, the generated chessboard images are placed on random backgrounds. The chessboard must be fully contained within the background, and the background edges are constrained to be no larger than 1/8 of the chessboard size, which seems to be reasonable from a practical point of view. Utilizing diverse backgrounds enhances the adaptability of the model to various real-world scenarios - the provided weights were trained with backgrounds randomly selected from the [Open Image Dataset](https://storage.googleapis.com/openimages/web/index.html).

- Possible future improvements: One might consider to generate even more realistic datasets. Here are some suggestions, which might be implemented in future - adding semi-transparent arrows on board (common analyst's tool) as they tend to confuse the network when used excessively, adding rank/file notation to the edge of the board, spawning mouse cursor on the image to avoid misinterpreting it with a white figure etc.

- Data augmentation: During the training, augmenting the training data is necessary. Even a simple augmentation like random brightness and contrast improves the model significantly, as, in practice, overly bright/dark boards were interpreted as full of pawns etc. We also add some random gauss noise to a small portion of the training data to make the network to focus on the features of the pieces rather than on the board pattern.

To train the model, we use the tensorflow implementation of the AdamW optimizer. Once per few epochs we generate new training dataset. To follow the training process we use two metrics - the obvious square accuracy is quite dubious, as there are too many easy examples (empty squares). In practice, the (almost) only relevant metric is the whole board accuracy, which we use as the second one.



