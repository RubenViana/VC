# VC_proj

## Task 1

- **Input:**
    - Image containing one or more LEGO bricks
- **Output:**
    - Total number of bricks
    - Position of the bricks (bb)
    - Number of different colours

### Image Pre-Processing
- Image resizing
- Image bluring

### Image Processing

#### Object Detection
- Applyed Canny edge detection
- Use Canny results to find contours
- Removed contours that have overlapped bounding boxes
- Remain contours are the target objects

#### Color Detection
- For each contours calculate the histogram to find the most used color
- Colors that have a 'distance' above the threshold are consider unique

### How to run

```
$ python3 ipp.py input.json output.json
```

- input.json -> file containing the list of images path
- output.json -> file where the results will be stored