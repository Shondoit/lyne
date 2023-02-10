# lyne
Python processing pipelines

Using generators, operations and proxy objects for convenient construction of pipelines.

```python
from lyne import *
from lyne.clip import *

target_size = (768, 768)
min_face_strength = 10

pipe = (
    list_dir.using(I.source_dir)
    | open_image
        | cond_size(I.image, min_size=target_size)
    | calc_focus
        | Cond(I.focus < 70, 'focus')
    | calc_lightness
        | Cond(I.lightness < 50, 'too_dark')
        | Cond(I.lightness > 205, 'too_bright')
    | calc_collage
        | Cond(I.collage >= 200, 'collage')

    | generate_attention('a photo of a face')
        | Cond(I.attention.max() <= 0, 'no_face')
        | scale_array(I.attention, (0, 10%Rel.pos), (0, 255), clip=True) >> I.attention
        | Cond(I.attention.mean() < min_face_strength, 'small_face')

    | generate_attention('a photo of multiple people')
        | Cond(I.attention.max() > 0, 'multi_people')

    | generate_attention('a photo of a person')
        | Cond(I.attention.max() <= 0, 'no_person')
        | scale_array(I.attention, (0, 10%Rel.pos), (0, 255), clip=True) >> I.attention
        | add_alpha_channel(I.attention)
    
    | change_dir.using(O.path, O.target_dir)
    | change_ext.using(O.path, O.target_ext)
    | save_skipped_image
    
    | purge_skipped
    | save_image
)

base_dir = r"D:\SD\training\Subject"
item = Item(
    source_dir=fr"{base_dir}\orig",
    target_dir=fr"{base_dir}\raw",
    target_ext='.png',
)
results = pipe.process(item)

#list() will iterate over the entire generator
list(results)
```
