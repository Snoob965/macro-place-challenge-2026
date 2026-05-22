from macro_place.benchmark import Benchmark
import torch
import numpy as np

class EmergencyShelfPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        n = benchmark.num_macros
        cw = benchmark.canvas_width
        ch = benchmark.canvas_height
        macro_sizes = benchmark.macro_sizes.numpy()
        placement = np.zeros((n, 2))
        
        fixed_mask = getattr(benchmark, 'macro_fixed', None)
        macro_positions = getattr(benchmark, 'macro_positions', None)
        
        placed_boxes = []
        if fixed_mask is not None and macro_positions is not None:
            macro_positions = macro_positions.numpy()
            placement[fixed_mask] = macro_positions[fixed_mask]
            for i in range(n):
                if fixed_mask[i]:
                    px, py = placement[i]
                    mw, mh = macro_sizes[i]
                    placed_boxes.append([px, py, px + mw, py + mh])
        else:
            fixed_mask = np.zeros(n, dtype=bool)

        movable_indices = np.where(~fixed_mask)[0]
        heights = macro_sizes[:, 1]
        sorted_movable = movable_indices[np.argsort(heights[movable_indices])[::-1]]

        x, y = 0.0, 0.0
        row_h = 0.0
        gap = 0.1

        for idx in sorted_movable:
            mw, mh = macro_sizes[idx]
            placed = False
            
            while y + mh <= ch:
                if x + mw > cw:
                    x = 0.0
                    y += row_h + gap
                    row_h = 0.0
                    continue
                
                conflict = None
                for b in placed_boxes:
                    if not (x + mw + gap <= b[0] or x >= b[2] + gap or y + mh + gap <= b[1] or y >= b[3] + gap):
                        conflict = b
                        break
                        
                if conflict is None:
                    placement[idx] = [x, y]
                    placed_boxes.append([x, y, x + mw, y + mh])
                    x += mw + gap
                    row_h = max(row_h, mh)
                    placed = True
                    break
                else:
                    x = conflict[2] + gap
                    
            if not placed:
                placement[idx] = [0.0, 0.0]

        return torch.tensor(placement, dtype=torch.float32)

class Placer(EmergencyShelfPlacer):
    pass