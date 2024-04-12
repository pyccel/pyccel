""" A file containing 2D analytical mappings to test Pyccel lambdify function.
"""

class PolarMapping:
    """
    Represents a Polar 2D Mapping object (Annulus).
    """
    expressions = {'x': 'c1 + (rmin*(1-x1)+rmax*x1)*cos(x2)',
                    'y': 'c2 + (rmin*(1-x1)+rmax*x1)*sin(x2)'}
    constants = {'rmin': 0.0, 'rmax': 1.0, 'c1' : 0.0, 'c2' : 0.0}

#==============================================================================
class TargetMapping:
    """
    Represents a Target 2D Mapping object.
    """
    expressions = {'x': 'c1 + (1-k)*x1*cos(x2) - D*x1**2',
                    'y': 'c2 + (1+k)*x1*sin(x2)'}
    constants = {'c1': 0.0, 'c2': 0.0, 'k' : 0.5, 'D' : '1.0'}

#==============================================================================
class CzarnyMapping:
    """
    Represents a Czarny 2D Mapping object.
    """
    expressions = {'x': '(1 - sqrt( 1 + eps*(eps + 2*x1*cos(x2)) )) / eps',
                    'y': 'c2 + (b / sqrt(1-eps**2/4) * x1 * sin(x2)) /'
                        '(2 - sqrt( 1 + eps*(eps + 2*x1*cos(x2)) ))'}
    constants = {'eps' : 0.1, 'c2' : 0.0, 'b' : 1.0}


