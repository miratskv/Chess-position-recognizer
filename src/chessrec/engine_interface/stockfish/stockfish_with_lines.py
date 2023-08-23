from stockfish import Stockfish

class StockfishLines(Stockfish):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_top_moves(self, num_top_moves: int = 5, lines_len: int = 6) -> list[dict]:
            """Returns info on the top lines in the position. Only difference between this function and the Stockfish.get_top_moves()
                is in "Move": current_line[current_line.index("pv") + 1:], instead of current_line[current_line.index("pv") + 1]
            Args:
                num_top_moves:
                    The number of lines to return info on, assuming there are at least
                    those many legal moves.
            Returns:
                A list of dictionaries. In each dictionary, there are keys for Move, Centipawn, and Mate;
                the corresponding value for either the Centipawn or Mate key will be None.
                If there are no moves in the position, an empty list is returned.
            """

            if num_top_moves <= 0:
                raise ValueError("num_top_moves is not a positive number.")
            old_MultiPV_value = self._parameters["MultiPV"]
            if num_top_moves != self._parameters["MultiPV"]:
                self._set_option("MultiPV", num_top_moves)
                self._parameters.update({"MultiPV": num_top_moves})
            self._go()
            lines = []
            while True:
                text = self._read_line()
                splitted_text = text.split(" ")
                lines.append(splitted_text)
                if splitted_text[0] == "bestmove":
                    break
            top_moves: List[dict] = []
            multiplier = 1 if ("w" in self.get_fen_position()) else -1
            for current_line in reversed(lines):
                if current_line[0] == "bestmove":
                    if current_line[1] == "(none)":
                        top_moves = []
                        break
                elif (
                    ("multipv" in current_line)
                    and ("depth" in current_line)
                    and current_line[current_line.index("depth") + 1] == self.depth
                ):
                    multiPV_number = int(current_line[current_line.index("multipv") + 1])
                    if multiPV_number <= num_top_moves:
                        has_centipawn_value = "cp" in current_line
                        has_mate_value = "mate" in current_line
                        if has_centipawn_value == has_mate_value:
                            raise RuntimeError(
                                "Having a centipawn value and mate value should be mutually exclusive."
                            )

                        top_moves.insert(
                            0,
                            {
                                "Move": current_line[current_line.index("pv") + 1:current_line.index("pv") + 1 + lines_len],
                                "Centipawn": int(current_line[current_line.index("cp") + 1])
                                * multiplier
                                if has_centipawn_value
                                else None,
                                "Mate": int(current_line[current_line.index("mate") + 1])
                                * multiplier
                                if has_mate_value
                                else None,
                            },
                        )
                else:
                    break
            if old_MultiPV_value != self._parameters["MultiPV"]:
                self._set_option("MultiPV", old_MultiPV_value)
                self._parameters.update({"MultiPV": old_MultiPV_value})
            return top_moves