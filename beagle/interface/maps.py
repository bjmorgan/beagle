from subprocess import check_output
import numpy as np

# Would this be better as a class? Then one object would manage a set of specific MAPS data.

def np_string( v ):
    return ' '.join( [ str(f) for f in v ] )

def ref_energy( filename='ref_energy.out' ):
    with open( filename ) as f:
        return float( f.readlines()[0] )

def predicted_energy( n_atoms, ref_eng, expansion, s='str.out', e='eci.out' ):
    energy = check_output( [ "corrdump", "-c", "-s={}".format( s ) , "-eci={}".format( e ) ] ).decode( 'utf-8' ).strip()
    return ( float( energy ) + ref_eng * n_atoms ) * expansion

def structure_to_str_out( structure, filename, cell_expansion ):
    with open( filename, 'w' ) as f:
        for v in structure.lattice.matrix:
            f.write( "{}\n".format( np_string( v / cell_expansion ) ) )
        f.write( "{:f} 0.0 0.0\n0.0 {:f} 0.0\n0.0 0.0 {:f}\n".format( *cell_expansion ) )
        for site in structure:
            f.write( "{} {}\n".format( np_string( site.frac_coords * cell_expansion ), site.species_string ) )

