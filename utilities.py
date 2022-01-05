def non_blank_lines(f):
    for ln in f:
        line = ln.rstrip()
        if line:
            yield line
