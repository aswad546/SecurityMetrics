"""
MIT License

Copyright (c) 2021, Sohail Habib

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

------------------------------------------------------------------------------------------------------------------------

Internal format parser
=====================
reads internal touch format and returns as a tuple of Header and user[dict]->session[dict]->activity[dict]
       ->direction[dict]->list of feature vectors
"""
import os
import sys
import numpy


def internal_touch_data_to_numpy_array(path, lim=1000):
    """reads internal touch format and returns as a tuple of Header and user[dict]->session[dict]->activity[dict]
       ->direction[dict]->list of feature vectors"""

    if not os.path.isdir(path):
        print("Path does not exist " + path)
        sys.exit(1)

    fvs = dict()
    hdr = None
    files = os.listdir(path)
    for fname in files[0:min(len(files), lim)]:
        with open(os.path.join(path, fname)) as f:
            lines = f.readlines()
        hdr = lines[0].split(',')[5:]
        for l in lines[1:]:
            toks = l.split(',')
            user, session, activity, direction = \
                map(lambda x: x.replace("'", ""), [toks[0], toks[1], toks[2], toks[4]])
            if user not in fvs:
                fvs[user] = dict()
            if session not in fvs[user]:
                fvs[user][session] = dict()
            if activity not in fvs[user][session]:
                fvs[user][session][activity] = dict()
            if direction not in fvs[user][session][activity]:
                fvs[user][session][activity][direction] = list()
            fvs[user][session][activity][direction].append(numpy.array(toks[5:], dtype=float))
    return hdr, fvs
