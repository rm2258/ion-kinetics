# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib as mpl # no matplotlib in inkscape python
import pickle as pkl

def network(states, prob, coords, adjacencyMatrix, 
            circleSize=6, circleAlpha=1, fontSize=2.5,
            awheadType=0, colorInto='#000000', colorOut='#000000', colorSelf='#000000'):
    """
    Function to draw nodes and attach connections between those nodes based
    on the adjacencyMatrix.

    Parameters
    ----------
    states : int or list or numpy array, (states,)
        The number of nodes to draw.
    prob : numpy array (states,)
        The probabilities of each state/node.
    coords : TYPE
        DESCRIPTION.
    adjacencyMatrix : numpy array (states, states)
        The connection matrix of the (i,j)-pairs.
    circleSize : TYPE, optional
        DESCRIPTION. The default is 6.
    circleAlpha : TYPE, optional
        DESCRIPTION. The default is 1.
    fontSize : TYPE, optional
        DESCRIPTION. The default is 2.5.
    awheadType : TYPE, optional
        DESCRIPTION. The default is 0.
    colorInto : TYPE, optional
        DESCRIPTION. The default is '#000000'.
    colorOut : TYPE, optional
        DESCRIPTION. The default is '#000000'.
    colorSelf : TYPE, optional
        DESCRIPTION. The default is '#000000'.

    Returns
    -------
    Layers : List
        List of layer objects.

    """
    debug = 1
    objs = drawCircles(states, prob, coords, circleSize=circleSize, fs=fontSize,
                       alpha=circleAlpha) # a list of group objects
    if isinstance(states, list) or isinstance(states, np.ndarray):
        states = len(states)
    if coords.shape == (states, 2):
        x_pos = coords[:, 0]
        y_pos = coords[:, 1]
    elif coords.shape == (2, states):
        x_pos = coords[0]
        y_pos = coords[1]
    x, y = np.where(adjacencyMatrix != 0) # make connection between pairs that 
                                          # are defined in matrix
    conList = [] # store all the connections for layer making at end
    for i in range(states):
        conList.append([]) # each node will have their own set of connections
    borderSizes = [] # store all the boder sizes to be added to each circle
    for i in range(states):
        borderSizes.append([0]*len(objs[i])) # each node will have their own 
                                             # set of borders
                                             # Is also padded with the total
                                             # amount of objects already in group
    if debug:
        print(x.shape)
        diffj = 0
        indexj = 0 # to avoid error in debug msg
    # maxBorders = 11 # only allow this many borders for any given node
    # curve = ['polyline', 'polyline', 'orthogonal']
    # strenght = [0, 25, 50, 75, 100]
    # curveSetting = 0
    for i, j in zip(x, y):
        strokeScale = adjacencyMatrix[i, j]
        desiredCSi = circleSize[i] + 0.8
        desiredCSj = circleSize[j] + 0.8 + (strokeScale / 0.1) * 0.3 # the offest is 
                                                               # based on the 
                                                               # length from the 
                                                               # placement of arrow
                                                               # head on the line
                                                               # to the tip
        # find closest circle size that would fit the next arrow connection
        indexi = np.abs(np.array(borderSizes[i]) - desiredCSi).argmin()
        diffi = np.abs(borderSizes[i][indexi] - desiredCSi) / desiredCSi
        # if diffi > 0.5  and len(borderSizes[i]) < maxBorders:
        if diffi > 0.22:
            # check that circle is not too small otherwise add a new circle
            newCirclei = circle((x_pos[i], y_pos[i]), desiredCSi,
                                conn_avoid=False, display='none', stroke='none')
            # now append new borders to the groups
            objs[i].append(newCirclei)
            indexi = len(objs[i]) - 1 # location of new circle in group
            borderSizes[i].append(desiredCSi)
        if i != j:
            indexj = np.abs(np.array(borderSizes[j]) - desiredCSj).argmin()
            diffj = np.abs(borderSizes[j][indexj] - desiredCSj) / desiredCSj
            if diffj > 0.09:
                newCirclej = circle((x_pos[j], y_pos[j]), desiredCSj,
                                    conn_avoid=False, display='none', stroke='none')
                objs[j].append(newCirclej)
                indexj = len(objs[j]) - 1 # location of new circle in group
                borderSizes[j].append(desiredCSj)
        if debug == 2 and (i == 0 or j == 0):
            print(diffi, diffj)
        if debug:
            print(i, j, len(objs[i]), len(objs[j]), indexi, indexj, 
                  len(borderSizes[i]), len(borderSizes[j]))
        if i < j:
            # out of state
            con = drawConnections(objs[i][indexi], objs[j][indexj],
                                  strokeScale=strokeScale,
                                  awheadType=awheadType, 
                                  c=colorOut, alpha=0.5, 
                                  curveType='polyline',
                                  curveStrength=0)
        elif i > j:
            # into state
            con = drawConnections(objs[i][indexi], objs[j][indexj],
                                  strokeScale=strokeScale,
                                  awheadType=awheadType,
                                  c=colorInto, alpha=0.5, 
                                  curveType='polyline',
                                  curveStrength=0)
        elif i == j:
            con = drawConnections(objs[i][0], objs[j][indexi], 
                                  strokeScale=strokeScale,
                                  awheadType=awheadType,
                                  c=colorSelf, alpha=0.5, 
                                  curveType='polyline', 
                                  curveStrength=0)
        # curveSetting += 1
        conList[i].append(con) # one connection arrow is added to each i state 
                               # to its j state pair
    # perform the layering
    Layers = []
    for i, obj in enumerate(all_shapes()):
        if obj.tag == 'g':
            newLayer = layer(f'State {i}')
            newLayer.append(obj)
            Layers.append(newLayer) # add all layers to the list whose index is
                                    # related to the state index
            if len(conList[i]):
                # only appends the arrows to this layer if this state has 
                # connections
                for con in conList[i]:
                    newLayer.append(con)
    return (Layers, objs)

def drawConnections(obj1, obj2, strokeScale, awheadType=1, c='#000000', 
                    alpha=1, curveType='orthogonal', curveStrength=0):
    """
    Helper function to add connection between two nodes
    Parameters
    ----------
    obj1 : TYPE
        DESCRIPTION.
    obj2 : TYPE
        DESCRIPTION.
    strokeScale : TYPE
        DESCRIPTION.
    awheadType : TYPE, optional
        DESCRIPTION. The default is 1.
    c : TYPE, optional
        DESCRIPTION. The default is '#000000'.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.
    curveType : str
        Either 'orthogonal' or 'polyline'.
    curveStrength : float
        Strenght of curving from 0 to 100.

    Returns
    -------
    obj : Object
        The Arrow object.

    """
    # there are two arrow head definitions in the following list more can be 
    # added
    if awheadType == 0:
        arrowhead = pointyArrowHead()
        m = marker(arrowhead, ref=(1,2), orient='auto-start-reverse')#, fill=c)
    else:
        arrowhead = roundedArrowHead()
        m = marker(arrowhead, ref=(0,0), orient='auto-start-reverse')#, fill=c)
    obj = connector(obj2, obj1, ctype=curveType, curve=curveStrength, 
                    stroke_width=strokeScale, 
                    stroke_linecap='round', marker_start=m, spacing=1,
                    stroke=c, stroke_opacity=alpha, opacity=alpha)
    return obj


def selfArrow(strokeScale, awheadType=1, c='#000000', alpha=1):
    """
        Helper function for drawing a self-arrow transition
        returns - Arrow Object
    """
    # Not setup to snap to the node
    # Also the arrow needs adjusting
    if awheadType == 0:
        arrowhead = pointyArrowHead()
        m = marker(arrowhead, ref=(1,2), orient='auto-start-reverse', fill=c)
    else:
        arrowhead = roundedArrowHead()
        m = marker(arrowhead, ref=(0,0), orient='auto-start-reverse', fill=c)
    obj = path([move(46.324845, 39.430272), 
                curve(0.0, 0.0, 9.772666, -18.193303, 20.767921, -11.398013), 
                curve(10.995255, 6.795289, -2.788919, 22.155939, -2.788919, 
                      22.155939)], opacity=alpha, stroke_width=strokeScale,
               stroke_linecap='round', stroke_opacity=alpha, marker_start=m,
               stroke=c)
    return obj

def pointyArrowHead():
    """
        Marker definition for Pointy arrow head
        returns - object
    """
    arrowhead = path([Move(0, 0), Line(4, 2), Line(0, 4), 
                      Curve(0, 4, 1, 3, 1, 2), Curve(1, 1, 0, 0, 0, 0), 
                      ZoneClose()], overflow='visible', fill_rule='evenodd',
                     fill='context-stroke', stroke='none')
    return arrowhead
    
def roundedArrowHead():
    """
        Maker definition for rounded arrow head
        returns - object
    """
    arrowhead = path([Move(5.77, 0.0), Line(-2.88, 5.0), Vert(-5.0), 
                      ZoneClose()], transform='scale(0.5, 0.5)', 
                     stroke='context-stroke', fill='context-stroke', 
                     fill_rule='evenodd', stroke_width='1pt')
    return arrowhead


def drawCircles(states, prob, coords, circleSize=6, fs=1, alpha=1, 
                strokeColor="#000000"):
    """
    Helper function for drawing the nodes on the canvas.
    Arguments:

    Parameters
    ----------
    states : TYPE
        DESCRIPTION.
    prob : TYPE
        DESCRIPTION.
    coords : numpy array (2,states) or (states, 2)
        The node coordinates.
    circleSize : TYPE, optional
        DESCRIPTION. The default is 6.
    fs : float, optional
        The font size of the probability text. The default is 1.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    objs : List
        List of group objects of length states.

    """
    debug = 0
    if debug:
        print(type(coords), states, coords.shape == (states, 2))
        x_pos = 1
    if isinstance(circleSize, int) or isinstance(circleSize, list):
        circleSize = np.ones(states) * circleSize
    if isinstance(states, int):
        numStates = states
        states = np.ones(states) * states
    else:
        numStates = len(states)
    if coords.shape == (numStates, 2):
        if debug:
            print("Here 1")
        x_pos = coords[:, 0]
        y_pos = coords[:, 1]
    elif coords.shape == (2, numStates):
        x_pos = coords[0]
        y_pos = coords[1]
    if debug:
        print(x_pos)
    objs = [] # add all the nodes to this list
    # avoidSetting = [False, False, True, False, False]
    for i in range(numStates):
        # the circles are placed relative to the origin of the canvas, which
        # can be changed with some simple inkscape property
        c = circle((x_pos[i], y_pos[i]), circleSize[i], opacity=alpha, 
                   stroke=strokeColor, stroke_width=0.3,
                   conn_avoid=False, stroke_opacity=alpha)
        # added offset to place the state number centered in circle
        # could also use inkscapes shape_inside argument for text to be place
        # inside the object, however I did not like how it looked
        # the default font is Calibri
        t1 = text(f'{states[i]}', (x_pos[i], y_pos[i] + 1.2), #shape_inside=c,
                 font_size=f'{fs}pt', text_align='center', text_anchor='middle', 
                 font_family='Calibri', _inkscape_font_specification='Calibri', 
                 fill="#ffffff")
        # added offset to place the probability of the state right below 
        # the node
        # also the font is set to be 20% smaller than the state font
        # t2 = text(f'{prob[i]:.4f}', (x_pos[i], y_pos[i] + circleSize[i] + 2.3),
        #           font_size=f'{fs * (1 - 0.4)}pt', text_align='center', 
        #           text_anchor='middle', font_family='Calibri', 
        #           _inkscape_font_specification='Calibri')
        # g = group([c, t1, t2], conn_avoid=False)
        g = group([c, t1], conn_avoid=False)
        objs.append(g)
    return objs

def drawOccupancy(state):
    """
    Function for drawing a microstate from an occupancy list
    Arguments:
                       
    Parameters
    ----------
    state : list
        The occupancy list of tokens arranged in the S0-S4 direction.

    Returns
    -------
    g : Group
        A group of objects.

    """
    pot_coords = [(3.199, 2.0), (3.199, 3.602), (3.199, 5.493), 
                  (3.199, 7.221), (3.199, 8.894), (3.199, 10.567), 
                  (3.199, 12.455), (3.199, 14.343), (3.199, 16.5)]
    wat_coords = [(2.883, 1.5), (2.883, 3.779), (2.883, 5.266), 
                  (2.883, 7.399), (2.883, 8.748), (2.883, 10.701), 
                  (2.883, 12.145), (2.883, 14.359), (2.883, 16.273)]
    indices = np.arange(9) # likely faster than overwriting the above coords
    if len(state) == 5:
        # for occupancy lists following the 5-binding site convention
        # i.e., no plannar binding sites
        indices = indices[::2]
    g = group([], conn_avoid=True) # to output a group of objects
    sf = drawFilter()
    g.append(sf)
    for index, token in zip(indices, state):
        if token == 'K':
            # then K+
            pot = drawPotassium()
            pot.translate(pot_coords[index])
            g.append(pot)
        elif token == 'W':
            # then water
            wat = drawWater()
            wat.translate(wat_coords[index])
            g.append(wat)
    # group center (4.1785, 10.7195)
    g.translate((-4.18, -10.72)) # re-center filter to origin of canvas
                                 # simplifies the translations
    return g
        

def drawFilter():
    """
        Helper function for drwaing an empty selectivity filter
        returns - group of objects
    """
    path632 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1, 
                   stroke_dasharray='none', stroke_opacity=1, 
                   paint_order='stroke markers fill')
    path633 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507, 
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.600001, 
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g633 = group([path632, path633], transform='translate(0, 0.38959)')
    path634 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1, 
                   stroke_dasharray='none', stroke_opacity=1, 
                   paint_order='stroke markers fill')
    path635 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507, 
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.600001, 
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g635 = group([path634, path635], transform='translate(4.83, 9.00387)')
    path636 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round', 
                   stroke_linejoin='round', stroke_miterlimit=6.1, 
                   stroke_dasharray='none', stroke_opacity=1, 
                   paint_order='stroke markers fill')
    path637 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507, 
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.600001, 
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g637 = group([path636, path637], transform='translate(4.83, 13.6749)')
    path638 = path([move(4.7359046, 25.596912), 
                    inkex.paths.line(-0.9530694, 1.54825)], 
                   stroke='#cfcfcf', fill='#bdbdbd', opacity=0.985507, 
                   fill_opacity=1, fill_rule='evenodd', stroke_width=1, 
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    path639 = path([move(4.7359046, 25.596912), 
                    inkex.paths.line(-0.9530694, 1.54825)], stroke='#ffffff', 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=0.6, 
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g639 = group([path638, path639], 
                 transform='matrix(0.394101 -0.919067 0.919067 0.394101 -17.7319 19.8618)')
    path640 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, 
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    path641 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507, 
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.600001,
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g641 = group([path640, path641], transform='translate(4.83, 19.0106)')
    path642 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1,
                   stroke_dasharray='none', stroke_opacity=1,
                   paint_order='stroke markers fill')
    path643 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507, 
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.600001,
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g643 = group([path642, path643], transform='translate(4.83, 4.86604)')
    path644 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1, 
                   stroke_dasharray='none', stroke_opacity=1, 
                   paint_order='stroke markers fill')
    path645 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507, 
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.600001,
                   stroke_linecap='round', stroke_linejoin='round',
                   stroke_miterlimit=6.1, stroke_dasharray='none',
                   stroke_opacity=1, paint_order='stroke markers fill')
    g645 = group([path644, path645], transform='translate(4.83, 0.38959)')
    path646 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1,
                   stroke_dasharray='none', stroke_opacity=1,
                   paint_order='stroke markers fill')
    path647 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)],
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507,
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.600001,
                   stroke_linecap='round', stroke_linejoin='round',
                   stroke_miterlimit=6.1, stroke_dasharray='none',
                   stroke_opacity=1, paint_order='stroke markers fill')
    g647 = group([path646, path647], transform='translate(0, 9.00387)')
    path648 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)],
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1,
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1, 
                   stroke_dasharray='none', stroke_opacity=1,
                   paint_order='stroke markers fill')
    path649 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507,
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.600001,
                   stroke_linecap='round', stroke_linejoin='round',
                   stroke_miterlimit=6.1, stroke_dasharray='none',
                   stroke_opacity=1, paint_order='stroke markers fill')
    g649 = group([path648, path649], transform='translate(0, 13.6749)')
    path650 = path([move(4.7359046, 25.596912), 
                    inkex.paths.line(-0.9530694, 1.54825)], stroke='#cfcfcf', 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1, 
                   stroke_dasharray='none', stroke_opacity=1, 
                   paint_order='stroke markers fill')
    path651 = path([move(4.7359046, 25.596912), 
                    inkex.paths.line(-0.9530694, 1.54825)], stroke='#ffffff', 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=0.6, 
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g651 = group([path650, path651], 
                 transform='matrix(0.988646 -0.150267 0.150267 0.988646 -3.79259 1.00229)')
    path652 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1, 
                   stroke_dasharray='none', stroke_opacity=1, 
                   paint_order='stroke markers fill')
    path653 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507,
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.600001,
                   stroke_linecap='round', stroke_linejoin='round',
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g653 = group([path652, path653], transform='translate(0, 19.0106)')
    path654 = path([Move(0.65423388, 0.54334677), Line(1.93, 1.6965726)], 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1, 
                   stroke_dasharray='none', stroke_opacity=1,
                   paint_order='stroke markers fill')
    path655 = path([Move(0.65423388, 0.54334677), Line(1.93, 1.6965726)], 
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507, 
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.6, 
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g655 = group([path654, path655], transform='matrix(-1 0 0 1 11.5442 1.21065)')
    path656 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1, 
                   stroke_dasharray='none', stroke_opacity=1, 
                   paint_order='stroke markers fill')
    path657 = path([Move(2.8297661, 6.5863575), Horz(4.7359046)], 
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507, 
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.600001, 
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g657 = group([path656, path657], transform='translate(0, 4.86604)')
    path658 = path([Move(0.65423388, 0.54334677), Line(1.93, 1.6965726)], 
                   fill='#bdbdbd', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1, 
                   stroke_dasharray='none', stroke_opacity=1, 
                   paint_order='stroke markers fill')
    path659 = path([Move(0.65423388, 0.54334677), Line(1.93, 1.6965726)], 
                   stroke='#ff0000', fill='#bdbdbd', opacity=0.985507, 
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.6, 
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g659 = group([path658, path659], transform='translate(0.899766, 1.21065)')
    path660 = path([Move(1.6295005, 1.6295005), Vert(24.317564)], 
                   fill='#c4c4c4', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1, 
                   stroke_dasharray='none', stroke_opacity=1, 
                   paint_order='stroke markers fill')
    path661 = path([Move(1.6295005, 1.6295005), Vert(24.317564)], 
                   stroke='#989898', fill='#c4c4c4', opacity=0.985507,
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.6,
                   stroke_linecap='round', stroke_linejoin='round', 
                   stroke_miterlimit=6.1, stroke_dasharray='none', 
                   stroke_opacity=1, paint_order='stroke markers fill')
    g661 = group([path660, path661], transform='translate(7.9625, 1.31115)')
    path662 = path([Move(1.6295005, 1.6295005), Vert(24.317564)], 
                   fill='#c4c4c4', opacity=0.985507, fill_opacity=1, 
                   fill_rule='evenodd', stroke_width=1, stroke_linecap='round',
                   stroke_linejoin='round', stroke_miterlimit=6.1,
                   stroke_dasharray='none', stroke_opacity=1, 
                   paint_order='stroke markers fill')
    path663 = path([Move(1.6295005, 1.6295005), Vert(24.317564)], 
                   stroke='#989898', fill='#c4c4c4', opacity=0.985507,
                   fill_opacity=1, fill_rule='evenodd', stroke_width=0.6,
                   stroke_linecap='round', stroke_linejoin='round',
                   stroke_miterlimit=6.1, stroke_dasharray='none',
                   stroke_opacity=1, paint_order='stroke markers fill')
    g663 = group([path662, path663], transform='translate(1.20027, 1.31115)')
    g = group([g633, g635, g637, g639, g641, g643, g645, g647, g649, g651, g653, 
               g655, g657, g659, g661, g663], 
              transform='matrix(0.80851 0 0 0.80851 22.0033 24.3273)', 
              display='inline')
    g.translate((-22.855, -25.341))
    return g

def drawPotassium():
    """
    Helper function for drawing a lone potassium ion

    Returns
    -------
    c : Object
        A shape object.

    """
    c = ellipse((76.884483, 72.982491), (1.0636501, 1.1304066), 
                stroke='#030004', fill='#c300e2', display='inline', 
                opacity=0.985507, fill_opacity=1, fill_rule='evenodd', 
                stroke_width=0.3, stroke_linecap='round', 
                stroke_linejoin='round', stroke_miterlimit=6.1, 
                stroke_dasharray='none', stroke_opacity=1,
                paint_order='stroke markers fill')
    c.scale(0.8)
    c.translate((-75.914, -71.958))
    return c

def drawWater():
    """
    Helper function for drawing a water molecule

    Returns
    -------
    g : Group object
        A group of objects.

    """
    ellipse664 = ellipse((8.8893547, 36.839996), (0.65193373, 0.69285023), 
                         stroke='#bdbdbd', fill='#ffffff', opacity=0.985507, 
                         fill_opacity=1, fill_rule='evenodd', 
                         stroke_width=0.367552, stroke_linecap='round', 
                         stroke_linejoin='round', stroke_miterlimit=6.1, 
                         stroke_dasharray='none', stroke_opacity=1,
                         paint_order='stroke markers fill')
    ellipse665 = ellipse((7.92835, 36.020687), (1.0636501, 1.1304066), 
                         stroke='#030004', fill='#ff0000', opacity=0.985507, 
                         fill_opacity=1, fill_rule='evenodd', stroke_width=0.3,
                         stroke_linecap='round', stroke_linejoin='round', 
                         stroke_miterlimit=6.1, stroke_dasharray='none', 
                         stroke_opacity=1, paint_order='stroke markers fill')
    ellipse666 = ellipse((7.0421195, 37.151432), (0.68834645, 0.73154825), 
                         stroke='#bdbdbd', fill='#ffffff', opacity=0.985507, 
                         fill_opacity=1, fill_rule='evenodd', 
                         stroke_width=0.388081, stroke_linecap='round', 
                         stroke_linejoin='round', stroke_miterlimit=6.1, 
                         stroke_dasharray='none', stroke_opacity=1, 
                         paint_order='stroke markers fill')
    g = group([ellipse664, ellipse665, ellipse666], 
              transform='translate(-1.73152, -18.5919)', 
              display='inline')
    g.scale(0.75)
    g.translate((-5.308, -21.208))
    return g


if 1:
    with open('E:\\uchicago\\remd_mthk\\amber\\no_res\\vol_0\\microstates_5site_package_rates.pkl', 'rb') as f:
        numStates, occuList, stationary_dist, rateMatrix, committor = pkl.load(f)
    f.close()
    with open('E:\\uchicago\\remd_mthk\\amber\\no_res\\vol_0\\network_coords.pkl', 'rb') as f:
        positions = pkl.load(f)
    f.close()
    with open("E:\\uchicago\\remd_mthk\\amber\\no_res\\vol_0\\fluxs_5bs_convention_0_7.pkl", 'rb') as f:
        major_flux = pkl.load(f)
    f.close()
    states = np.tile(np.arange(numStates), 5)
    # occuList = occuList*5 # copied 5 times because five blocks
    # cg_map = {0:4, 1:2, 2:17, 3:16, 4:3, 5:1, 6:13, 7:15, 8:0, 9:6, 10:5}
    stationary_dist = np.tile(stationary_dist, 5) # five blocks
    cs = (np.exp(stationary_dist / stationary_dist.max() * 1.4) + 0.25) * 2
    # a = np.zeros((130,130)) # for no connections
    if 0:
        # for MSM/rate contant connections
        transitionMatrix *= 10 # to make into ns^-1
        cond = transitionMatrix < 10**-10 # a tolerance instead of exactly zero
        # cond = transitionMatrix == 0
        transitionMatrix[cond == False] = np.exp(transitionMatrix[cond == False] \
                                                 - transitionMatrix[cond == False].max()) + 0.25
        transitionMatrix[cond] = 0.0
        # x,y = np.where(transitionMatrix != 0)
        # for i,j in zip(x,y):
        #     if (i >= states*2 and i < states*3) or (j >= states*2 and j < states*3):
        #         continue
        #     else:
        #         transitionMatrix[i,j] = 0
        # print(transitionMatrix, x.shape)
    if 0:
        # for rates connection
        cond = rateMatrix < 10**-10
        rateMatrix[cond == False] = np.exp(rateMatrix[cond == False] \
                                           - rateMatrix[cond == False].max()) + 0.0
        rateMatrix[cond] = 0.0
        x,y = np.where(rateMatrix != 0)
        for i,j in zip(x,y):
            if (i >= states*2 and i < states*3) or (j >= states*2 or j < states*3):
                continue
            else:
                rateMatrix[i,j] = 0
    if 1:
        # for flux
        major_flux *= 10
        cond = major_flux == 0
        # net_flux[cond == False] = np.exp(net_flux[cond == False]) - 0.5
        major_flux[cond == False] = np.exp(major_flux[cond == False] \
                                           - major_flux[cond == False].min()) + 0.0
        # print(major_flux)
    layers, objs = network(states, stationary_dist, positions, major_flux,
                           circleSize=cs, circleAlpha=1, colorInto='#000000', 
                           colorOut='#000000', colorSelf='#de00fa')
    colors = ['#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000',
              '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000',
              '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000',
              '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000',
              '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000',
              '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000',
              '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000',
              '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff0000',
              '#ff0000', '#ff0000', '#ff0000', '#ff0000', '#ff3636', '#1a1aff',
              '#1e1eff', '#ff0000', '#1c1cff', '#1e1eff', '#ff3434', '#ff3030',
              '#0000ff', '#0000ff', '#0000ff', '#ff1212', '#ff0a0a', '#ffdede',
              '#ff0000', '#ffc2c2', '#ffacac', '#ff0c0c', '#ff0c0c', '#bcbcff',
              '#ffa6a6', '#e6e6ff', '#ffaaaa', '#ff0000', '#ffc2c2', '#ffacac',
              '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff',
              '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff',
              '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff',
              '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff',
              '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff',
              '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff',
              '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff',
              '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff', '#0000ff',
              '#0000ff', '#0000ff', '#0000ff', '#0000ff']
    for i, color in enumerate(colors):
        objs[i][0].style(stroke='#000000', fill=color)
    Offset = [0, 2.9]
    for i in range(numStates*2, numStates*3):
        # wrap with modulo
        interDraw = drawOccupancy(occuList[i % numStates].split())
        # interDraw = drawOccupancy(occuList[cg_map[i]].split())
        # if i == 56:
        #     x,y = positions[i] + [cs[i], 0] + [0.6, 0]
        # else:
        #     x,y = positions[i] + [cs[i], 0] + Offset
        x,y = positions[i] + Offset
        interDraw.scale(1 - 0.6, (0,0))
        interDraw.rotate(90, (0,0))
        interDraw.translate((x, y)) # the translate method just adds to
                                    # the current position of the object. 
                                    # As for groups I think
                                    # it adds to the sub-objects, which
                                    # would be the same as adding to the 
                                    # center of that group
        # layers[i].append(interDraw)
        objs[i].append(interDraw)
    
