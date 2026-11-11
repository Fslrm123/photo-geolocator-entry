# Find where a photo was taken — code-only geolocator

A make.fun `photo-geolocator-01` entry that places each photo with its own
code alone: no model calls, no API key, no network beyond the trial env.

## How it works

`main.py` (Python 3.13, standard library only):

1. Reads the brief at `MAKEFUN_PROMPT_FILE`, lists the instance's 25 photos
   from the env (`GET /photos`), downloads each JPEG, and posts one
   latitude/longitude per photo (`POST /answers`).
2. Each JPEG is decoded by a small pure-stdlib baseline-DCT decoder that
   reads only the DC coefficient of every 8x8 block (and skips the AC
   coefficients), giving one average colour per block — about 114x64 colours
   for a 910x512 photo, in ~0.15 s per photo.
3. Simple visual cues are read off those block colours: snow on the ground,
   red dirt, yellow centre lines, lush tropical green, green reaching into
   the sky band, overcast sky, sand, dry grass, overall brightness and
   saturation.
4. Eleven world regions are scored from the cues; the answer is the
   softmax-weighted centre of the regions, blended 40% toward the photo
   pool's geographic centre (57, 27), because the cues are weak and a
   confident wrong region must not fling the answer to the far side of the
   world. Every photo is always answered: an unanswered photo counts the
   maximum error, so even a rough guess beats leaving it blank.

Fully deterministic: no clock, no randomness, requests in photo order, so a
replay leaves the same answers.

## Practice results (kit's own env + scorer, all 3 practice trials)

| trial | total error | median error | within 2500 km | score |
|------:|------------:|-------------:|---------------:|------:|
| 0 | 244,482 km | 10,504 km | 8% | 255,895 |
| 1 | 139,010 km | 4,286 km | 28% | 361,367 |
| 2 | 136,616 km | 5,655 km | 36% | 363,761 |

Mean score 327,008. The cues carry real but modest signal: snow reliably
means high latitude, red dirt and yellow lines help sometimes, and most
photos are too ambiguous for pixel colours alone to place closely. A vision
model would do far better; this entry is the honest code-only baseline.

## Run locally

```sh
python3 main.py  # with MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE set
```

## License

MIT — see `LICENSE`.
