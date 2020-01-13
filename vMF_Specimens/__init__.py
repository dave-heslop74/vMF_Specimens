import numpy as np
import pandas as pd
import qgrid
import ipywidgets as ipyw
from traitlets import traitlets

def A95toKR(alpha):
    return 2.99573 / (1-np.cos(np.deg2rad(alpha)))

def A3(k):
    return (1/np.tanh(k)-1/k)

def A3inv(r,N):
    return (3*(N**2*r)-r**3) / (N**3-N*r**2)

def a95(R,N):
    return np.arccos(1-(N-R)/R*(20**(1/(N-1))-1))/np.pi*180

def MADtoA95(MAD,n,anchor):
    n0 = np.array([3,4,5,6,7,8,9,10,11,12,13,14,15,16,100])
    
    if anchor==False:
        CMAD0 = np.array([7.69,3.90,3.18,2.88,2.71,2.63,2.57,2.54,2.51,2.48,2.46,2.44,2.43,2.43,2.37])
    else:
        CMAD0 = np.array([6.0,5.0,4.63,4.43,4.31,4.24,4.18,4.14,4.12,4.11,4.08,4.08,4.06,4.05,3.99])
    
    return np.interp(n,n0,CMAD0)*MAD


###############

def input_frame(X):

    input_df = pd.DataFrame({
        'Dec [deg.]' : pd.Series([np.nan],index=list(range(1)),dtype='float32'),
        'Inc [deg.]' : pd.Series([np.nan],index=list(range(1)),dtype='float32'),
        'n' : pd.Series([0],index=list(range(1)),dtype='int'),
        'MAD [deg.]' : pd.Series([np.nan],index=list(range(1)),dtype='float32'),
        'Anchored' : pd.Series(False,index=list(range(1)),dtype='bool')
    })
    
    #input_df['Anchored'] = input_df['n'] == 0
    input_widget = qgrid.show_grid(input_df, show_toolbar=True)
    input_widget.layout = ipyw.Layout(width='65%')

    #X = {}
    X['input_widget'] = input_widget
    display(X['input_widget'])
    
    return X


def read_data(X):
    
    final_df = X['input_widget'].get_changed_df()
    X['dec'] = np.array(final_df['Dec [deg.]'])
    X['inc'] = np.array(final_df['Inc [deg.]'])
    X['mad'] = np.array(final_df['MAD [deg.]'])
    X['n'] = np.array(final_df['n'])
    X['anchor'] = np.array(final_df['Anchored'])
    X['N'] = len(X['dec'])
    
    return X

def process_data(X):

    X = read_data(X)
    
    alpha95 = np.empty(X['N'])
    for i in range(X['N']):
        alpha95[i] = MADtoA95(X['mad'][i],X['n'][i],X['anchor'][i])
    
    rho = A3(A95toKR(alpha95))

    MU=[]
    for i in range(X['N']):
        temp=dir2cart((X['dec'][i],X['inc'][i]))
        MU.append(np.squeeze(temp))    
    A = np.sum(rho[:,np.newaxis] * MU,axis=0)
    A0 = np.sum(MU,axis=0)

    R = np.linalg.norm(A)
    R0 = np.linalg.norm(A0)

    MU = A/R
    MU0 = A0/R0

    X['mu*'] = cart2dir(MU)[0:2]
    X['a95*'] = (a95(R,X['N']))

    X['mu'] = cart2dir(MU0)[0:2]
    X['a95'] = (a95(R0,X['N']))
    
    # Print results
    spacer = ipyw.HTML(value='<font color="white">Spacer text</font>')
    title_main = ipyw.HTML(value='<h3>Results</h3>')
    hr = ipyw.HTML(value='<hr style="height:2px;border:none;color:#333;background-color:#333;" />')
    title_star = ipyw.HTML(value='<h4>Incorperating uncertainty</h4>')
    inc_star = ipyw.HTMLMath(value='Inc$^*$ [deg.] = {0:.2f}'.format(X['mu*'][1]))
    dec_star = ipyw.HTMLMath(value='Dec$^*$ [deg.] = {0:.2f}'.format(X['mu*'][0]))
    a95_star = ipyw.HTMLMath(value=r'$\alpha 95^*$ [deg.] = {0:.2f}'.format(X['a95*']))
    title_null = ipyw.HTML(value='<h4>Without uncertainty</h4>')
    inc_null = ipyw.HTMLMath(value='Inc [deg.] = {0:.2f}'.format(X['mu'][1]))
    dec_null = ipyw.HTMLMath(value='Dec [deg.] = {0:.2f}'.format(X['mu'][0]))
    a95_null = ipyw.HTMLMath(value=r'$\alpha 95$ [deg.] = {0:.2f}'.format(X['a95']))
    results_star = ipyw.VBox((title_star,inc_star,dec_star,a95_star))
    results_null = ipyw.VBox((title_null,inc_null,dec_null,a95_null))
    results_comb = ipyw.HBox((results_star,spacer,results_null))
    results = ipyw.VBox((hr,title_main,results_comb,hr))
    display(results)
    
    return X

    #### Buttons

class LoadedButton(ipyw.Button):
    """A button that can holds a value as a attribute."""

    def __init__(self, value=None, *args, **kwargs):
        super(LoadedButton, self).__init__(*args, **kwargs)
        # Create the value attribute.
        self.add_traits(value=traitlets.Any(value))


def process(ex):    
    process_data(ex.value)


    #### Functions from pmagpy

def dir2cart(d):
    """
    Converts a list or array of vector directions in degrees (declination,
    inclination) to an array of the direction in cartesian coordinates (x,y,z)
    Parameters
    ----------
    d : list or array of [dec,inc] or [dec,inc,intensity]
    Returns
    -------
    cart : array of [x,y,z]
    Examples
    --------
    >>> pmag.dir2cart([200,40,1])
    array([-0.71984631, -0.26200263,  0.64278761])
    """
    ints = np.ones(len(d)).transpose(
    )  # get an array of ones to plug into dec,inc pairs
    d = np.array(d)
    rad = np.pi/180.
    if len(d.shape) > 1:  # array of vectors
        decs, incs = d[:, 0] * rad, d[:, 1] * rad
        if d.shape[1] == 3:
            ints = d[:, 2]  # take the given lengths
    else:  # single vector
        decs, incs = np.array(float(d[0])) * rad, np.array(float(d[1])) * rad
        if len(d) == 3:
            ints = np.array(d[2])
        else:
            ints = np.array([1.])
    cart = np.array([ints * np.cos(decs) * np.cos(incs), ints *
                     np.sin(decs) * np.cos(incs), ints * np.sin(incs)]).transpose()
    return cart


def cart2dir(cart):
    """
    Converts a direction in cartesian coordinates into declination, inclinations
    Parameters
    ----------
    cart : input list of [x,y,z] or list of lists [[x1,y1,z1],[x2,y2,z2]...]
    Returns
    -------
    direction_array : returns an array of [declination, inclination, intensity]
    Examples
    --------
    >>> pmag.cart2dir([0,1,0])
    array([ 90.,   0.,   1.])
    """
    cart = np.array(cart)
    rad = np.pi/180  # constant to convert degrees to radians
    if len(cart.shape) > 1:
        Xs, Ys, Zs = cart[:, 0], cart[:, 1], cart[:, 2]
    else:  # single vector
        Xs, Ys, Zs = cart[0], cart[1], cart[2]
    if np.iscomplexobj(Xs):
        Xs = Xs.real
    if np.iscomplexobj(Ys):
        Ys = Ys.real
    if np.iscomplexobj(Zs):
        Zs = Zs.real
    Rs = np.sqrt(Xs**2 + Ys**2 + Zs**2)  # calculate resultant vector length
    # calculate declination taking care of correct quadrants (arctan2) and
    # making modulo 360.
    Decs = (np.arctan2(Ys, Xs) / rad) % 360.
    try:
        # calculate inclination (converting to degrees) #
        Incs = np.arcsin(Zs/Rs) / rad
    except:
        print('trouble in cart2dir')  # most likely division by zero somewhere
        return np.zeros(3)

    return np.array([Decs, Incs, Rs]).transpose()  # return the directions list


    

