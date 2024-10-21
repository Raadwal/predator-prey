import PIL
import IPython
import base64
import imageio

import numpy as np

def save_video(frames, video_filename):
  with imageio.get_writer(video_filename + '.mp4', fps=30) as video:
    for frame in frames:
      img = PIL.Image.fromarray(frame)
      img = img.resize((608, 400))  # Resize to dimensions divisible by 16
      video.append_data(np.array(img))

def show_video(video_filename):
  video = open(video_filename, 'rb').read()
  b64 = base64.b64encode(video)

  tag = """
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>""".format(b64.decode())

  return IPython.display.HTML(tag)