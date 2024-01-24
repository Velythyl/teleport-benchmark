import numpy as np
from PIL import Image
from jax import numpy as jp


def load_ascii(ascii_path):
    with open(ascii_path, "r") as f:
        f.readline()
        w = int(f.readline().split(" ")[-1])
        h = int(f.readline().split(" ")[-1])
        f.readline()
        acsii_grid = f.read()

    PASSABLE = [".", "G", "S"]
    IMPASSABLE = ["T", "W"]
    OUT_OF_BOUNDS = ["@", "O"]
    for char in PASSABLE:
        acsii_grid = acsii_grid.replace(char, "0")
    for char in IMPASSABLE:
        acsii_grid = acsii_grid.replace(char, "1")
    for char in OUT_OF_BOUNDS:
        acsii_grid = acsii_grid.replace(char, "2")
    acsii_grid = acsii_grid.split("\n")[:-1]

    import numpy as np
    # jax is way too slow here
    grid = np.zeros((h,w))
    for i, row in enumerate(acsii_grid):
        for j, char in enumerate(row):
            grid[i,j] = int(char)

    grid = jp.array(grid)
    img = jp.array(grid)
    grid = grid.at[grid > 0].set(1)

    return img, grid.T

def load_png(png_path):
    img = Image.open( png_path )
    white_bg = Image.new("RGBA", img.size, "WHITE")
    white_bg.paste(img, (0,0), img)
    white_bg = white_bg.convert("L")
    data = np.asarray( white_bg, dtype="int32" )

    passable = data > 255/2
    impassable = data < 255/2

    data[passable] = 0
    data[impassable] = 1

    max_hw = np.max(data.shape[:-1])
    final_image = np.ones((max_hw, max_hw))
    final_image[0:data.shape[0], 0:data.shape[1]] = data

    return final_image, final_image.T


if __name__ == "__main__":
    load_png("../maps/costmap_full_room.png")