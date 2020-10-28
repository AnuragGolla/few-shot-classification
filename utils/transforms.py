"""
Tensor transformations utilized in models.
"""

def unstack(x, top2dims):
    xsz = x.size()
    nsz = top2dims
    if len(xsz) > 1:
        nsz += [xsz[-1]]
    return x.view(nsz)


