# Lyne
Python processing pipelines
---
Using generators, operations and proxy objects for convenient construction of pipelines.

Lyne is a way to easily write a processing pipeline.
Combine operations, add conditions and process any stream of input data.

Lyne uses a few classes to make the definitions easy to write and understand.

## Item, Stream
An `Item` is a convenience class inheriting a dictionary. You can access the values by using the keys as attributes. `item.key` will simply return `item['key']`. Another convenience is that it will return `None` for missing keys. A special field is `skip`; this value will define if the item should be skipped from further processing. It can be any truthy value, including a textual reason for skipping that can be used for later filtering or debugging.

A `Stream` is simply a wrapper around an iterator. It's only used to differentiate between iterable arguments and actual streams of data that need to be processed.

## Operation
An `Operation` is a wrapper around a function that is performed on an `Item`. The `Operation` itself accepts a `Stream` and will output a `Stream`. (The first in a pipeline can also accept an `Item`)

The arguments to an `Operation` will be placeholders. Only during processing will these placeholders be filled with the actual data needed for the wrapped function.

## Proxy
A `Proxy` is a placeholder that accepts all kinds of (simple) operations and aplies them at a later time on an actual object. For example: `placeholder = ItemProxy().value * 10 + 5` will remember the operations applied to it. Later you can call `obj.value = 3; Proxy.apply(placeholder, obj)` and it will return 35. (First it retrieves the value of obj.value, then multiplies by 10 and lastly adds 5 to it.)

`Proxy` objects are used for the placeholders of `Operation` objects; both as arguments as well as outputs. It makes it easy to write `open_image(I.path) >> I.image`. When defining this `Operation` it will store the placeholders. When processing the pipeline, for each item that is being processed the placeholder will be replaced by the actual value of the item. `>>` defines the output location, so in this example an image will be opened using item.path, and saved to item.image.

## RelativeValue
`RelativeValue` are used in places where you want to specify a value in relation to another. Want to shrink an image to 50%? Threshold a grayscale image at 10%? Rescale an array where the values between 10% and 50% map to 0 and 255? `RelativeValue` will make this easy.

Note that the function must be aware of these relative values and apply min-max to get an absolute value. 
The package modules take advantage of this where possible. 

## Modules

### fs
This is the basic filesystem module. It contains simple path manipulation.

### util
Assistance module that contains basic functions.

### cv
OpenCV based image module, used for image calculations, manipulation, detection, etc.

### clip
CLIPText based module. Used for image captioning, segmentation and other cool CLIP stuff.

### depth
MiDaS based depth module. Used to generate depth maps.

## Convenience functions
The core module contains a number of convenience function to make  writing pipelines easier and more readable.

### Op, Cond

`Op` is a convenience object for creating a new `Operation`. If you write `op_print = Op[print]` it will wrap the print function into an `Operation`. Afterwards you can use this `Operation` by calling it with placeholders like so: `op_print(I.value_to_print, ':', I.other_value * 10)`. 

Or you can combine them all in one line: `Op[print](I.value)` (The square brackets around the function instead of parentheses have been deliberately chosen for readability).

The right shift operator (>>) is used to conveniently define the output mapping. `Op[max](I.val_a, I.val_b) >> I.result` will calculate the max of the two input values and store the result into the item's `result` attribute.

`Cond` is a simple `Operation` that will store the second argument in `item.skip` if the first argument is true. It makes it easy to say `Cond(I.value <= 0, 'value too low')`; if item.value is zero or lower, the item will be skipped from future processing. The reason will be 'value too low'.

### S, I, O
S, I and O are used to define Item, Stream and Output proxies, respectively.

If you want to specify a placeholder in an `Operation`, you can easily do so by writing `I.attribute` or `I['field']`. `S` will be replaced by the current Stream and `I` will be replaced by the current Item.

`O` is a special case and defines a mapping for the output. If the function returns a dictionary with multiple keys and you want to extract only one of them you can write `... >> {I.result: O['key_to_extract']}`.

### Rel
`Rel` is a convenience object to easily define a `RelativeValue`. You can write `rel_val = 50%Rel` and it will create a new `RelativeValue` with 0.5 as the value. If you then convert it to an absolute value by calling `Value.to_abs(rel_val, 0, 200)` it will return 100 (50% of 200).

Special cases exist for `Rel.pos` and `Rel.neg`. This will return the percentage of the positive range. If you have an array with values ranging from -100 to +50, then `50%Rel` is -25 (exactly in the middle of -100 and +50; `50%Rel.pos` is +25 (in the middle of the positive range: 0 to 50); and `50%Rel.neg` is -50 (in the middle of the negative range: -100 to 0).

# Example

Below is a complete example of a pipeline.

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
    
    | change_dir.using(I.path, I.target_dir)
    | change_ext.using(I.path, I.target_ext)
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
