def import_distribution(model):

    list_distribution = {}
    if model=='Hubbard':
        list_distribution['3_1'] = {(0,0,0):1, (-1,0,1):1, (1,0,1):1}
        list_distribution['5_1'] = {(0,0,0):1, (-1,0,1):1, (1,0,1):1, (0,-1,1):1, (0,1,1):1}
        list_distribution['6_1'] = {(0,0,0):2, (-1,0,1):1, (1,0,1):1, (0,-1,1):1, (0,1,1):1}
        list_distribution['7_1'] = {(0,0,0):3, (-1,0,1):1, (1,0,1):1, (0,-1,1):1, (0,1,1):1}
        list_distribution['8_1'] = {(0,0,0):4, (-1,0,1):1, (1,0,1):1, (0,-1,1):1, (0,1,1):1}
        list_distribution['9_1'] = {(0,0,0):5, (-1,0,1):1, (1,0,1):1, (0,-1,1):1, (0,1,1):1}
        list_distribution['10_1'] = {(0,0,0):2, (-1,0,1):1, (1,0,1):1, (0,-1,1):1, (0,1,1):1, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1}
        list_distribution['11_1'] = {(0,0,0):3, (-1,0,1):1, (1,0,1):1, (0,-1,1):1, (0,1,1):1, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1}
        list_distribution['12_1'] = {(0,0,0):4, (-1,0,1):1, (1,0,1):1, (0,-1,1):1, (0,1,1):1, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1}
        list_distribution['12_2'] = {(0,0,0):4, (-1,0,1):2, (1,0,1):2, (0,-1,1):2, (0,1,1):2}
        list_distribution['13_1'] = {(0,0,0):4, (-1,0,1):1, (1,0,1):1, (0,-1,1):1, (0,1,1):1, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1}
        list_distribution['14_1'] = {(0,0,0):2, (-1,0,1):2, (1,0,1):2, (0,-1,1):2, (0,1,1):2, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1}
        list_distribution['15_1'] = {(0,0,0):3, (-1,0,1):2, (1,0,1):2, (0,-1,1):2, (0,1,1):2, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1}
        list_distribution['16_1'] = {(0,0,0):4, (-1,0,1):2, (1,0,1):2, (0,-1,1):2, (0,1,1):2, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1}
        list_distribution['20_1'] = {(0,0,0):4, (-1,0,1):2, (1,0,1):2, (0,-1,1):2, (0,1,1):2, (-1,-1,0):2, (-1,1,0):2, (1,-1,0):2, (1,1,0):2}
        list_distribution['21_1'] = {(0,0,0):5, (-1,0,1):3, (1,0,1):3, (0,-1,1):3, (0,1,1):3, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1}
        list_distribution['24_1'] = {(0,0,0):4, (-1,0,1):3, (1,0,1):3, (0,-1,1):3, (0,1,1):3, (-1,-1,0):2, (-1,1,0):2, (1,-1,0):2, (1,1,0):2}
        list_distribution['25_1'] = {(0,0,0):5, (-1,0,1):3, (1,0,1):3, (0,-1,1):3, (0,1,1):3, (-1,-1,0):2, (-1,1,0):2, (1,-1,0):2, (1,1,0):2}  # best
        list_distribution['25_2'] = {(0,0,0):5, (-1,0,1):4, (1,0,1):4, (0,-1,1):4, (0,1,1):4, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1} 
        list_distribution['25_3'] = {(0,0,0):5, (-1,0,1):3, (1,0,1):3, (0,-1,1):3, (0,1,1):3, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1, (-2,0,0):1, (2,0,0):1, (0,-2,0):1, (0,2,0):1}
        list_distribution['26_1'] = {(0,0,0):6, (-1,0,1):3, (1,0,1):3, (0,-1,1):3, (0,1,1):3, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1, (-2,0,0):1, (2,0,0):1, (0,-2,0):1, (0,2,0):1}
        list_distribution['29_1'] = {(0,0,0):5, (-1,0,1):4, (1,0,1):4, (0,-1,1):4, (0,1,1):4, (-1,-1,0):3, (-1,1,0):3, (1,-1,0):3, (1,1,0):3}
        list_distribution['30_1'] = {(0,0,0):6, (-1,0,1):4, (1,0,1):4, (0,-1,1):4, (0,1,1):4, (-1,-1,0):3, (-1,1,0):3, (1,-1,0):3, (1,1,0):3}

    elif model=='spinless':
        list_distribution['5_1'] = {(-1,):2, (0,):1, (1,):2}
        list_distribution['6_1'] = {(-1,):2, (0,):2, (1,):2}
        list_distribution['7_1'] = {(-1,):2, (0,):3, (1,):2}
        list_distribution['8_1'] = {(-2,):1, (-1,):2, (0,):2, (1,):2, (2,):1}
        list_distribution['8_2'] = {(-1,):2, (0,):4, (1,):2}
        list_distribution['9_1'] = {(-2,):1, (-1,):2, (0,):3, (1,):2, (2,):1} 
        list_distribution['10_1'] = {(-2,):1, (-1,):2, (0,):4, (1,):2, (2,):1} #check
        list_distribution['10_2'] = {(-2,):1, (-1,):3, (0,):2, (1,):3, (2,):1}
        list_distribution['11_1'] = {(-2,):1, (-1,):3, (0,):3, (1,):3, (2,):1}
        list_distribution['11_2'] = {(-2,):1, (-1,):2, (0,):5, (1,):2, (2,):1}
        list_distribution['12_1'] = {(-2,):1, (-1,):3, (0,):4, (1,):3, (2,):1}  # good
        list_distribution['12_2'] = {(-2,):1, (-1,):4, (0,):2, (1,):4, (2,):1}
        list_distribution['13_1'] = {(-2,):1, (-1,):4, (0,):3, (1,):4, (2,):1}
        list_distribution['13_2'] = {(-2,):1, (-1,):3, (0,):5, (1,):3, (2,):1}
        list_distribution['14_1'] = {(-2,):1, (-1,):4, (0,):4, (1,):4, (2,):1}
        list_distribution['14_2'] = {(-2,):1, (-1,):3, (0,):6, (1,):3, (2,):1}
        list_distribution['15_1'] = {(-2,):1, (-1,):4, (0,):5, (1,):4, (2,):1}
        list_distribution['15_2'] = {(-2,):1, (-1,):5, (0,):3, (1,):5, (2,):1}
        list_distribution['16_1'] = {(-2,):1, (-1,):5, (0,):4, (1,):5, (2,):1}
        list_distribution['16_2'] = {(-2,):1, (-1,):4, (0,):6, (1,):4, (2,):1}
        list_distribution['17_1'] = {(-2,):1, (-1,):5, (0,):5, (1,):5, (2,):1}
        list_distribution['17_2'] = {(-2,):1, (-1,):4, (0,):7, (1,):4, (2,):1}
        list_distribution['18_1'] = {(-2,):1, (-1,):6, (0,):4, (1,):6, (2,):1}
        list_distribution['18_2'] = {(-2,):1, (-1,):5, (0,):6, (1,):5, (2,):1}
        list_distribution['18_3'] = {(-2,):1, (-1,):4, (0,):8, (1,):4, (2,):1}   # check
        list_distribution['19_1'] = {(-2,):1, (-1,):6, (0,):5, (1,):6, (2,):1}
        list_distribution['19_2'] = {(-2,):2, (-1,):5, (0,):5, (1,):5, (2,):2}
        list_distribution['19_3'] = {(-2,):2, (-1,):4, (0,):7, (1,):4, (2,):2}   #check
        list_distribution['20_1'] = {(-2,):1, (-1,):7, (0,):4, (1,):7, (2,):1}
        list_distribution['20_2'] = {(-2,):2, (-1,):6, (0,):4, (1,):6, (2,):2}
        list_distribution['20_3'] = {(-2,):2, (-1,):5, (0,):6, (1,):5, (2,):2}   # check
        list_distribution['20_4'] = {(-2,):2, (-1,):4, (0,):8, (1,):4, (2,):2}   # check
        list_distribution['20_5'] = {(-2,):1, (-1,):4, (0,):10, (1,):4, (2,):1}   # check
        list_distribution['21_1'] = {(-2,):2, (-1,):4, (0,):9, (1,):4, (2,):2}   # check
        list_distribution['24_1'] = {(-2,):2, (-1,):6, (0,):8, (1,):6, (2,):2} 
        list_distribution['25_1'] = {(-2,):2, (-1,):6, (0,):9, (1,):6, (2,):2}
        list_distribution['26_1'] = {(-2,):2, (-1,):6, (0,):10, (1,):6, (2,):2}
        list_distribution['29_1'] = {(-2,):2, (-1,):7, (0,):11, (1,):7, (2,):2}
        list_distribution['30_1'] = {(-2,):3, (-1,):7, (0,):10, (1,):7, (2,):3}
       
    return list_distribution
