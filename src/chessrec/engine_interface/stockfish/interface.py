from stockfish import StockfishException
from .stockfish_with_lines import StockfishLines as Stockfish

class EngineInterface():
    def __init__(self, elo, hash_mb, threads, depth):
        self.elo = elo
        self.hash_mb = hash_mb
        self.threads = threads
        self.depth = depth
        self.engine = self.reset(
            self.elo, 
            self.depth, 
            self.hash_mb, 
            self.threads
        )

    def reset(self, elo, depth, hash_mb, threads):
        engine = Stockfish()
        engine.set_depth(depth)
        engine.set_elo_rating(elo)
        engine.update_engine_parameters({"Hash": hash_mb, "Threads": threads})
        return engine

    def get_top_lines(self, FEN):
        try:
            self.engine.set_fen_position(FEN)
            best_lines = self.engine.get_top_moves(3, lines_len = 6)
            return best_lines
        except StockfishException as e:
            return None

    def stockfish_evaluation(self, FEN):
        top_lines = self.get_top_lines(FEN)
        if top_lines:
            text_report = self.text_report_lines(top_lines)
        else:
            self.engine = self.reset(
                self.elo,
                self.depth,
                self.hash_mb,
                self.threads
            )
            text_report = "Error: Illegal position\n"
        return text_report

    def text_report_lines(self, lines):
        text_report = "Engine moves:\n"
        for move in lines:
            if move['Centipawn'] != None:
                score = f'{move["Centipawn"]/100:.2f}' 
            else:
                score = f'M {move["Mate"]}'
            text_report += f'{score}: {*move["Move"],}\n'
        return text_report 
