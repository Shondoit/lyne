import sys
import os
# Add lyne package from relative dir
_script_dir = os.path.dirname(os.path.abspath(__file__))
_lyne_dir = os.path.dirname(_script_dir)
sys.path.append(_lyne_dir)

import argparse
from lyne import *
from lyne.clip import *


def main(args):
    pipe = (
        # Load all images from source dir
        list_dir.using(args.input_dir, extensions=['.jpg', '.png'])
        | progress
        | open_image
    )

    if args.mask_type == 'attention':
        pipe |= (
            # Filter and focus on person
            generate_attention(args.attention_prompt)
            | scale_array(I.attention, (0, 10%Rel.pos), (0, 255), clip=True) >> I.attention
            | add_alpha_channel(I.attention)
        )
    elif args.mask_type == 'gradient':
        pipe |= (
            create_gradient(exp=args.gradient_exp, max_val=255)
            | add_alpha_channel(I.gradient)
        )
    elif args.mask_type == 'depth':
        raise NotImplementedError()

    pipe |= (
        # Change path and save processed image
        change_dir.using(I.path, args.output_dir)
        | change_ext.using(I.path, '.png')
        | save_image(overwrite=args.overwrite)
    )

    list(pipe.process())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True, help='Source directory, where the images are loaded from.')
    parser.add_argument('--output-dir', type=str, required=True, help='Target directory, where the images are saved to.')
    parser.add_argument('--mask-type', type=str, choices=['attention', 'gradient'], default='attention', help='Mask type used. Attention will use CLIP to highlight "a photo of a person". Gradient will apply a polynomial function to focus on the center.')
    parser.add_argument('--attention-prompt', type=str, default='a photo of a person', help='Prompt used for the CLIP attention map.')
    parser.add_argument('--gradient-exp', type=float, default=3, help='Exponent used for gradient map.')
    parser.add_argument('--overwrite', type=bool, default=False, help='Wether to overwrite the images.')

    args = parser.parse_args()
    main(args)
