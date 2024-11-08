import imageio
from IPython.display import Image, display

def gif(data,gif_path,fps):
    frames = []
    ch = data.shape[0]
    for i in range(ch):
        img = data[0]["image"][i, 0]
        img = img - img.min()
        img = img/img.max()
        img_rgb = (img * 255).astype('uint8')
        frames.append(img_rgb)
    imageio.mimsave(gif_path, frames, fps=fps)  # Adjust fps as needed
    display(Image(filename=gif_path))
