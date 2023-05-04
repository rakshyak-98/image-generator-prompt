import torch
from diffusers import StableDiffusionPipeline

from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok

import base64
from io import BytesIO

app = Flask(__name__)
run_with_ngrok(app)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
pipe.to('cuda')

@app.route('/')
def initial():
    return render_template('index.html')

@app.route('/submit-prompt', methods=['POST'])
def generate_image():
    prompt = request.form['prompt-input']
    print(f'Generating an image of {prompt}')

    image = pipe(prompt).images[0]
    print(f'output image: {image}')
    print('Image generated! Converting image to png...')

    buffered = BytesIO()
    image.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue())
    img_str = 'data:image/png;base64,' + str(img_str)[2:-1]
    print(f'Created image string: {img_str}')

    print('Sending image...')
    return render_template('index.html', generated_image=img_str)


if __name__ == '__main__':
    app.run()