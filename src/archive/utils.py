import numpy as np
import scipy.io
import os
from scipy.stats import norm
from scipy.stats import multivariate_normal


def rollpad_1D(array, roll, mode='minimum'):

  # from https://gist.github.com/Gridflare
    """Analogous to numpy.roll but using np.pad instead for greater flexibility"""
    if roll > 0:
        xpad = (roll,0)
        xtrim = slice(0,-roll)
    else:
        xpad = (0,-roll)
        xtrim = slice(-roll, None)
    
    padded = np.pad(array, (xpad), mode=mode)
    trimmed = padded[xtrim]
    assert trimmed.shape == array.shape
    return trimmed


def create_prior_slice_10(width_idx, shift, true_slice):
    
    width_data = np.array([[0,0,0,0.5,1,0.5,0,0,0,0],[0,0,0,0.5,1,1,0.5,0,0,0],[0,0,0.5,1,1,1,0.5,0,0,0],[0,0,0.5,1,1,1,1,0.5,0,0],
                         [0,0.5,1,1,1,1,1,0.5,0,0]])
    
    prior_slice = width_data[width_idx]
    
    
    if width_idx==2 or width_idx==3:
      roll_val = np.argmax(true_slice)-(np.argmax(prior_slice)+1)

    elif width_idx==4:
      roll_val = np.argmax(true_slice)-(np.argmax(prior_slice)+2)

    else:
      roll_val = np.argmax(true_slice)-np.argmax(prior_slice)

    #print(roll_val)
    prior_slice = rollpad_1D(prior_slice,roll_val)
    
    
    if shift ==0:
      val_shift = np.array([0,0])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    elif shift==2:
      if np.argmax(true_slice)<2:
        val_shift = np.array([2,2])
      elif np.argmax(true_slice)>7:
        val_shift = np.array([-2,-2])
      else:
        val_shift = np.array([-2,2])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
     
    elif shift==4:
      if np.argmax(true_slice)<4:
        val_shift = np.array([4,4])
      elif np.argmax(true_slice)>5:
        val_shift = np.array([-4,-4])
      else:
        val_shift = np.array([-4,4])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    elif shift==6:

      if np.argmax(true_slice)==4:
        val_shift = np.array([5,5])
      elif np.argmax(true_slice)==5:
        val_shift = np.array([-5,-5])
      elif np.argmax(true_slice)<6:
        val_shift = np.array([6,6])
      elif np.argmax(true_slice)>3:
        val_shift = np.array([-6,-6])
      
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    else: #('max')

      if width_idx==2 or width_idx==3:
        if np.argmax(true_slice)<5:
          shift_idx = 9-np.argmax(true_slice)#-np.argmax(prior_slice)
          shift_idx = shift_idx # -1

        elif np.argmax(true_slice)>=5:
          shift_idx = 0-np.argmax(true_slice)
          shift_idx = shift_idx +1


      elif width_idx==4:
        if np.argmax(true_slice)<5:
          shift_idx = 9-np.argmax(true_slice)#-np.argmax(prior_slice)
          shift_idx = shift_idx# -2

        elif np.argmax(true_slice)>=5:
          shift_idx = 0-np.argmax(true_slice)
          shift_idx = shift_idx +2

      else:
        if np.argmax(true_slice)<5:
          shift_idx = 9-np.argmax(true_slice)

        elif np.argmax(true_slice)>=5:
          shift_idx = 0-np.argmax(true_slice)

    prior_slice = rollpad_1D(prior_slice, shift_idx)
    
    return prior_slice



def create_prior_slice_4(width_idx, shift, true_slice):
    
    width_data = np.array([[0.5,1,0.5,0],[0.5,1,1,0.5],[0.5,1,1,0.5]])
    
    shift_2d_val =  [0,1,2] # 3 =='max' for size of 4
    shift_2d = shift_2d_val[shift]
    
    prior_slice = width_data[width_idx]
    
    
    # if width_idx==2: # or width_idx==4 :
    #   roll_val = np.argmax(true_slice)-(np.argmax(prior_slice)+1)

    #else:
    roll_val = np.argmax(true_slice)-np.argmax(prior_slice)

    #print(roll_val)
    prior_slice = rollpad_1D(prior_slice,roll_val)
    
    
    if shift_2d ==0:
      val_shift = np.array([0,0])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
        
    elif shift_2d==1:
      if np.argmax(true_slice)<1:
        val_shift = np.array([1,1])
      elif np.argmax(true_slice)>2:
        val_shift = np.array([-1,-1])
      else: # 2
        val_shift = np.array([-1,1])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    elif shift_2d==2:
      if np.argmax(true_slice)<2:
        val_shift = np.array([2,2])
      elif np.argmax(true_slice)>2:
        val_shift = np.array([-2,-2])
      else: # 2
        val_shift = np.array([-2,2])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
     
    elif shift_2d==3: # 3 si possible sinon 2
      if np.argmax(true_slice)==0:
        val_shift = np.array([3,3])
      elif np.argmax(true_slice)==3:
        val_shift = np.array([-3,-3])
      else: # shift of 2
        if np.argmax(true_slice)<2:
            val_shift = np.array([2,2])
        elif np.argmax(true_slice)>2:
            val_shift = np.array([-2,-2])
        else: # 2
            val_shift = np.array([-2,2])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]


    prior_slice = rollpad_1D(prior_slice, shift_idx)
    
    return prior_slice



def create_prior_slice_8(width_idx, shift, true_slice):
    
    width_data = np.array([[0,0,0.5,1,0.5,0,0,0],[0,0,0.5,1,1,0.5,0,0],
                         [0,0.5,1,1,1,1,0.5,0]])
    
    shift_2d_val =  [0,2,4] # 3 =='max' for size of 4
    shift_2d = shift_2d_val[shift]
        
    prior_slice = width_data[width_idx]
    
    
    if width_idx==2:
      roll_val = np.argmax(true_slice)-(np.argmax(prior_slice)+1)

    # elif width_idx==4:
    #   roll_val = np.argmax(true_slice)-(np.argmax(prior_slice)+2)

    else:
      roll_val = np.argmax(true_slice)-np.argmax(prior_slice)

    #print(roll_val)
    prior_slice = rollpad_1D(prior_slice,roll_val)
    
    
    if shift_2d ==0:
      val_shift = np.array([0,0])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    elif shift_2d==2:
      if np.argmax(true_slice)<2:
        val_shift = np.array([2,2])
      elif np.argmax(true_slice)>5:
        val_shift = np.array([-2,-2])
      else:
        val_shift = np.array([-2,2])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
     
    elif shift_2d==4:
      if np.argmax(true_slice)<4:
        val_shift = np.array([4,4])
      elif np.argmax(true_slice)>3:
        val_shift = np.array([-4,-4])
      else:
        val_shift = np.array([-4,4])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    elif shift_2d==6:
      if np.argmax(true_slice)==0 or np.argmax(true_slice)==1:
        val_shift = np.array([6,6])
      elif np.argmax(true_slice)==6 or np.argmax(true_slice)==7:
        val_shift = np.array([-6,-6])
      elif np.argmax(true_slice)==2:
        val_shift = np.array([5,5])
      elif np.argmax(true_slice)==5:
        val_shift = np.array([-5,-5])
      elif np.argmax(true_slice)==3:
          val_shift = np.array([4,4])
      elif np.argmax(true_slice)==4:
          val_shift = np.array([-4,-4])    
      
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    else: # 7 si possible
        
      if np.argmax(true_slice)==0:
        val_shift = np.array([7,7])
      elif np.argmax(true_slice)==1:
        val_shift = np.array([6,6])
      elif np.argmax(true_slice)==7:
        val_shift = np.array([-7,-7])
      elif np.argmax(true_slice)==6:
        val_shift = np.array([-6,-6])
      elif np.argmax(true_slice)==2:
        val_shift = np.array([5,5])
      elif np.argmax(true_slice)==5:
        val_shift = np.array([-5,-5])
      elif np.argmax(true_slice)==3:
          val_shift = np.array([4,4])
      elif np.argmax(true_slice)==4:
          val_shift = np.array([-4,-4]) 
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    prior_slice = rollpad_1D(prior_slice, shift_idx)
    
    return prior_slice


def create_prior_slice_4_new(width_idx, shift, id_max):
        
    #width_data = np.array([0.5,1,0.5,0])
    shift_2d_val =  [0,1,1,2] # 3 =='max' for size of 4
    shift_2d = shift_2d_val[shift]
    prior_slice = np.zeros(4)
    width_2d_val=[1,1,1]
    width_2d = width_2d_val[width_idx]
    
    if shift_2d ==0:
      val_shift = np.array([0,0])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
        
    elif shift_2d==1:
      if id_max<1:
        val_shift = np.array([1,1])
      elif id_max>2:
        val_shift = np.array([-1,-1])
      else: # 2
        val_shift = np.array([-1,1])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    elif shift_2d==2:
      if id_max<2:
        val_shift = np.array([2,2])
      elif id_max>=2:
        val_shift = np.array([-2,-2])
      #else: # 2
      #  val_shift = np.array([-2,2])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
    
    if width_2d==1:
        prior_slice[int(id_max+shift_idx)] = 1
        if (id_max+shift_idx-1) >= 0:
            prior_slice[int(id_max+shift_idx-1)] = 0.5
        if (id_max+shift_idx+1) < 4:
            prior_slice[int(id_max+shift_idx+1)] = 0.5
            
    return prior_slice





def create_prior_slice_8_new(width_idx, shift, id_max):
    
    #width_data = np.array([[0,0,0.5,1,0.5,0,0,0],[0,0,0.5,1,1,0.5,0,0],
                         #[0,0.5,1,1,1,1,0.5,0]])
    
    shift_2d_val =  [0,1,2,3] # 3 =='max' for size of 4
    shift_2d = shift_2d_val[shift]
    
    width_2d_val=[1,2,4]
    width_2d = width_2d_val[width_idx]
        
    prior_slice = np.zeros(8)
    
    
    if shift_2d ==0:
      val_shift = np.array([0,0])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
        
    elif shift_2d==1:
      if id_max<1:
        val_shift = np.array([1,1])
      elif id_max>6:
        val_shift = np.array([-1,-1])
      else:
        val_shift = np.array([-1,1])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    elif shift_2d==2:
      if id_max<2:
        val_shift = np.array([2,2])
      elif id_max>5:
        val_shift = np.array([-2,-2])
      else:
        val_shift = np.array([-2,2])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
     
    elif shift_2d==3:
      if id_max<3:
        val_shift = np.array([3,3])
      elif id_max>4:
        val_shift = np.array([-3,-3])
      else:
        val_shift = np.array([-3,3])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
    
    
    if width_2d==1:
        prior_slice[int(id_max+shift_idx)] = 1
        if (id_max+shift_idx-1) >= 0:
            prior_slice[int(id_max+shift_idx-1)] = 0.5
        if (id_max+shift_idx+1) < 8:
            prior_slice[int(id_max+shift_idx+1)] = 0.5
        
    elif width_2d==2:
        
        prior_slice[int(id_max+shift_idx)] = 1
        
        if (id_max+shift_idx-1)<0:
            
            if (id_max+shift_idx+1) < 8:
                prior_slice[int(id_max+shift_idx+1)] = 1
            if (id_max+shift_idx+2) < 8:
                prior_slice[int(id_max+shift_idx+2)] = 0.5
            if (id_max+shift_idx-1) >= 0:
                prior_slice[int(id_max+shift_idx-1)] = 0.5
        
        elif (id_max+shift_idx+1)>=8:
            if (id_max+shift_idx-1) >= 0:
                prior_slice[int(id_max+shift_idx-1)] = 1
            if (id_max+shift_idx-2) >= 0:
                prior_slice[int(id_max+shift_idx-2)] = 0.5
            if (id_max+shift_idx+1) < 8:
                prior_slice[int(id_max+shift_idx+1)] = 0.5
                    
        else:
        
            p_prob = np.random.rand()
            
            if p_prob < 0.5:
                if (id_max+shift_idx-1) >= 0:
                    prior_slice[int(id_max+shift_idx-1)] = 1
                if (id_max+shift_idx-2) >= 0:
                    prior_slice[int(id_max+shift_idx-2)] = 0.5
                if (id_max+shift_idx+1) < 8:
                    prior_slice[int(id_max+shift_idx+1)] = 0.5

            else:
                if (id_max+shift_idx+1) < 8:
                    prior_slice[int(id_max+shift_idx+1)] = 1
                if (id_max+shift_idx+2) < 8:
                    prior_slice[int(id_max+shift_idx+2)] = 0.5
                if (id_max+shift_idx-1) >= 0:
                    prior_slice[int(id_max+shift_idx-1)] = 0.5
                
    elif width_2d==4:
        
        prior_slice[int(id_max+shift_idx)] = 1        
        
        if (id_max+shift_idx-2)<0:
            
            if (id_max+shift_idx+1) < 8:
                prior_slice[int(id_max+shift_idx+1)] = 1
            if (id_max+shift_idx+2) < 8:
                prior_slice[int(id_max+shift_idx+2)] = 1
            if (id_max+shift_idx+3) < 8:
                prior_slice[int(id_max+shift_idx+3)] = 0.5
            if (id_max+shift_idx-1) >= 0:
                prior_slice[int(id_max+shift_idx-1)] = 1
            if (id_max+shift_idx-2) >= 0:
                prior_slice[int(id_max+shift_idx-2)] = 0.5
                    
                    
        elif (id_max+shift_idx+2) >= 8: 
            if (id_max+shift_idx-1) >= 0:
                prior_slice[int(id_max+shift_idx-1)] = 1
            if (id_max+shift_idx-2) >= 0:
                prior_slice[int(id_max+shift_idx-2)] = 1
            if (id_max+shift_idx-3) >= 0:
                prior_slice[int(id_max+shift_idx-3)] = 0.5
            if (id_max+shift_idx+1) < 8:
                prior_slice[int(id_max+shift_idx+1)] = 1
            if (id_max+shift_idx+2) < 8:
                prior_slice[int(id_max+shift_idx+2)] = 0.5
        
        else:
            
            p_prob = np.random.rand()
        
            if p_prob < 0.5:
                if (id_max+shift_idx-1) >= 0:
                    prior_slice[int(id_max+shift_idx-1)] = 1
                if (id_max+shift_idx-2) >= 0:
                    prior_slice[int(id_max+shift_idx-2)] = 1
                if (id_max+shift_idx-3) >= 0:
                    prior_slice[int(id_max+shift_idx-3)] = 0.5
                if (id_max+shift_idx+1) < 8:
                    prior_slice[int(id_max+shift_idx+1)] = 1
                if (id_max+shift_idx+2) < 8:
                    prior_slice[int(id_max+shift_idx+2)] = 0.5

            else:
                if (id_max+shift_idx+1) < 8:
                    prior_slice[int(id_max+shift_idx+1)] = 1
                if (id_max+shift_idx+2) < 8:
                    prior_slice[int(id_max+shift_idx+2)] = 1
                if (id_max+shift_idx+3) < 8:
                    prior_slice[int(id_max+shift_idx+3)] = 0.5
                if (id_max+shift_idx-1) >= 0:
                    prior_slice[int(id_max+shift_idx-1)] = 1
                if (id_max+shift_idx-2) >= 0:
                    prior_slice[int(id_max+shift_idx-2)] = 0.5

    return prior_slice


def create_prior_slice_4_gaus(width_idx, shift, id_max):
        
    #width_data = np.array([0.5,1,0.5,0])
    shift_2d_val =  [0,1,1,2] # 3 =='max' for size of 4
    shift_2d = shift_2d_val[shift]
    prior_slice = np.zeros(4)
    width_2d_val=[0,1,2]
    width_2d = width_2d_val[width_idx]
    
    if shift_2d ==0:
      val_shift = np.array([0,0])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
        
    elif shift_2d==1:
      if id_max<1:
        val_shift = np.array([1,1])
      elif id_max>2:
        val_shift = np.array([-1,-1])
      else: # 2
        val_shift = np.array([-1,1])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    elif shift_2d==2:
      if id_max<2:
        val_shift = np.array([2,2])
      elif id_max>=2:
        val_shift = np.array([-2,-2])
      #else: # 2
      #  val_shift = np.array([-2,2])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
    
    if width_2d==0:

        prior_slice = norm.pdf(np.linspace(0,3,4), id_max+shift_idx, 1.0)
        prior_slice = (prior_slice-np.min(prior_slice))/(np.max(prior_slice)-np.min(prior_slice))
        
    elif width_2d==1:

        prior_slice = norm.pdf(np.linspace(0,3,4), id_max+shift_idx, 1.5)
        prior_slice = (prior_slice-np.min(prior_slice))/(np.max(prior_slice)-np.min(prior_slice))
        
    elif width_2d==2:

        prior_slice = norm.pdf(np.linspace(0,3,4), id_max+shift_idx, 2.0)
        prior_slice = (prior_slice-np.min(prior_slice))/(np.max(prior_slice)-np.min(prior_slice))

    return prior_slice

def create_prior_slice_8_gaus(width_idx, shift, id_max):
    
    #width_data = np.array([[0,0,0.5,1,0.5,0,0,0],[0,0,0.5,1,1,0.5,0,0],
                         #[0,0.5,1,1,1,1,0.5,0]])
    
    shift_2d_val =  [0,1,3,'max'] # 3 =='max' for size of 4
    shift_2d = shift_2d_val[shift]
    
    width_2d_val=[1,2,4]
    width_2d = width_2d_val[width_idx]
        
    prior_slice = np.zeros(8)
    
    
    if shift_2d ==0:
      val_shift = np.array([0,0])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
        
    elif shift_2d==1:
      if id_max<1:
        val_shift = np.array([1,1])
      elif id_max>6:
        val_shift = np.array([-1,-1])
      else:
        val_shift = np.array([-1,1])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    elif shift_2d==2:
      if id_max<2:
        val_shift = np.array([2,2])
      elif id_max>5:
        val_shift = np.array([-2,-2])
      else:
        val_shift = np.array([-2,2])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
     
    elif shift_2d==3:
      if id_max<3:
        val_shift = np.array([3,3])
      elif id_max>4:
        val_shift = np.array([-3,-3])
      else:
        val_shift = np.array([-3,3])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
    
    elif shift_2d=='max':
      if id_max<4:
        val_shift = int(7-id_max)
      elif id_max>=4:
        val_shift = int(-id_max)
      #else:
        #val_shift = np.array([-4,4])
      shift_idx = val_shift
    
    
    if width_2d==1:
        prior_slice = norm.pdf(np.linspace(0,7,8), id_max+shift_idx, 1.0)
        prior_slice = (prior_slice-np.min(prior_slice))/(np.max(prior_slice)-np.min(prior_slice))
        
    elif width_2d==2:
        
        prior_slice = norm.pdf(np.linspace(0,7,8), id_max+shift_idx, 2.0)
        prior_slice = (prior_slice-np.min(prior_slice))/(np.max(prior_slice)-np.min(prior_slice))
                
    elif width_2d==4:
        
        prior_slice = norm.pdf(np.linspace(0,7,8), id_max+shift_idx, 4.0)
        prior_slice = (prior_slice-np.min(prior_slice))/(np.max(prior_slice)-np.min(prior_slice))        


    return prior_slice



def create_prior_slice_2D_gaus(width_idx, shift, id_max):
    
    #width_data = np.array([[0,0,0.5,1,0.5,0,0,0],[0,0,0.5,1,1,0.5,0,0],
                         #[0,0.5,1,1,1,1,0.5,0]])
    
    shift_2d_val =  [0,1,3,'max'] # 3 =='max' for size of 4
    shift_2d = shift_2d_val[shift]
    
    width_2d_val=[1]
    width_2d = width_2d_val[width_idx]
            
    x, y = np.mgrid[0:8:1, 0:4:1]
    pos = np.dstack((x, y))
    
    
    if shift_2d ==0:
        
        rv = multivariate_normal([id_max[0], id_max[1]],  allow_singular=True)
        prior = rv.pdf(pos)
        prior = (prior-np.min(prior))/(np.max(prior)-np.min(prior))
        
        return prior
      
    elif shift_2d==1:
      if id_max[0]<1:
        val_shift_8 = np.array([1,1])
      elif id_max[0]>6:
        val_shift_8 = np.array([-1,-1])
      else:
        val_shift_8 = np.array([-1,1])
      shift_idx_8 = np.random.choice(val_shift_8, 1, p=[0.5, 0.5])[0]
    
      if id_max[1]<1:
        val_shift_4 = np.array([1,1])
      elif id_max[1]>2:
        val_shift_4 = np.array([-1,-1])
      else: # 2
        val_shift_4 = np.array([-1,1])
      shift_idx_4 = np.random.choice(val_shift_4, 1, p=[0.5, 0.5])[0]
  

     
    elif shift_2d==3:
      if id_max[0]<3:
        val_shift_8 = np.array([3,3])
      elif id_max[0]>4:
        val_shift_8 = np.array([-3,-3])
      else:
        val_shift_8 = np.array([-3,3])
      shift_idx_8 = np.random.choice(val_shift_8, 1, p=[0.5, 0.5])[0]
    
      if id_max[1]<1:
        val_shift_4 = np.array([1,1])
      elif id_max[1]>2:
        val_shift_4 = np.array([-1,-1])
      else: # 2
        val_shift_4 = np.array([-1,1])
      shift_idx_4 = np.random.choice(val_shift_4, 1, p=[0.5, 0.5])[0]
    
    elif shift_2d=='max':
      if id_max[0]<4:
        val_shift_8 = int(7-id_max[0])
      elif id_max[0]>=4:
        val_shift_8 = int(-id_max[0])
      #else:
        #val_shift = np.array([-4,4])
      shift_idx_8 = val_shift_8
    
      if id_max[1]<2:
        val_shift_4 = np.array([2,2])
      elif id_max[1]>=2:
        val_shift_4 = np.array([-2,-2])
      #else: # 2
      #  val_shift = np.array([-2,2])
      shift_idx_4 = np.random.choice(val_shift_4, 1, p=[0.5, 0.5])[0]
    
    
    rv = multivariate_normal([id_max[0]+shift_idx_8, id_max[1]+shift_idx_4],  allow_singular=True)
    prior = rv.pdf(pos)
    prior = (prior-np.min(prior))/(np.max(prior)-np.min(prior))
        
#     elif width_2d==2:
        
#         prior_slice = norm.pdf(np.linspace(0,9,10), id_max+shift_idx, 2.0)
#         prior_slice = (prior_slice-np.min(prior_slice))/(np.max(prior_slice)-np.min(prior_slice))
                
#     elif width_2d==4:
        
#         prior_slice = norm.pdf(np.linspace(0,9,10), id_max+shift_idx, 4.0)
#         prior_slice = (prior_slice-np.min(prior_slice))/(np.max(prior_slice)-np.min(prior_slice))        


    return prior


def create_prior_slice_10_gaus(width_idx, shift, id_max):
    
    #width_data = np.array([[0,0,0.5,1,0.5,0,0,0],[0,0,0.5,1,1,0.5,0,0],
                         #[0,0.5,1,1,1,1,0.5,0]])
    
    shift_2d_val =  [0,2,4,'max'] # 3 =='max' for size of 4
    shift_2d = shift_2d_val[shift]
    
    width_2d_val=[1,2,4]
    width_2d = width_2d_val[width_idx]
        
    prior_slice = np.zeros(10)
    
    
    if shift_2d ==0:
      val_shift = np.array([0,0])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
        
    elif shift_2d==1:
      if id_max<1:
        val_shift = np.array([1,1])
      elif id_max>8:
        val_shift = np.array([-1,-1])
      else:
        val_shift = np.array([-1,1])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]

    elif shift_2d==2:
      if id_max<2:
        val_shift = np.array([2,2])
      elif id_max>7:
        val_shift = np.array([-2,-2])
      else:
        val_shift = np.array([-2,2])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
     
    elif shift_2d==3:
      if id_max<3:
        val_shift = np.array([3,3])
      elif id_max>6:
        val_shift = np.array([-3,-3])
      else:
        val_shift = np.array([-3,3])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
    
    elif shift_2d==4:
      if id_max<4:
        val_shift = np.array([4,4])
      elif id_max>5:
        val_shift = np.array([-4,-4])
      else:
        val_shift = np.array([-4,4])
      shift_idx = np.random.choice(val_shift, 1, p=[0.5, 0.5])[0]
    
    
    elif shift_2d=='max':
      if id_max<5:
        val_shift = int(9-id_max)
      elif id_max>=5:
        val_shift = int(-id_max)
      #else:
        #val_shift = np.array([-4,4])
      shift_idx = val_shift
    
    
    if width_2d==1:
        prior_slice = norm.pdf(np.linspace(0,9,10), id_max+shift_idx, 1.0)
        prior_slice = (prior_slice-np.min(prior_slice))/(np.max(prior_slice)-np.min(prior_slice))
        
    elif width_2d==2:
        
        prior_slice = norm.pdf(np.linspace(0,9,10), id_max+shift_idx, 2.0)
        prior_slice = (prior_slice-np.min(prior_slice))/(np.max(prior_slice)-np.min(prior_slice))
                
    elif width_2d==4:
        
        prior_slice = norm.pdf(np.linspace(0,9,10), id_max+shift_idx, 4.0)
        prior_slice = (prior_slice-np.min(prior_slice))/(np.max(prior_slice)-np.min(prior_slice))        


    return prior_slice



def unflatten(map_flat, ch2xy, dim):
    
    the_map =  np.zeros(dim)
    for i in range(len(map_flat)):
        the_map[int(ch2xy[i,0]-1),int(ch2xy[i,1]-1)] = map_flat[i]
    
    return the_map



def load_matlab_data(path_to_dataset,dataset,m_i): 
    # The order of the variables inside the dict changes between Macaque, Cebus and rats
    if dataset=='nhp':
        if m_i==0:
            Cebus1_M1_190221 = scipy.io.loadmat(path_to_dataset+'/Cebus1_M1_190221.mat')
            Cebus1_M1_190221= {'emgs': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][0][0],
           'nChan': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][2][0][0],
           'sorted_isvalid': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][8],
           'sorted_resp': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][9],
           'sorted_respMean': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][10],
           'ch2xy': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][16]}
            SET=Cebus1_M1_190221
        if m_i==1:
            Cebus2_M1_200123 = scipy.io.loadmat(path_to_dataset+'/Cebus2_M1_200123.mat')  
            Cebus2_M1_200123= {'emgs': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][0][0],
           'nChan': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][2][0][0],
           'sorted_isvalid': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][8],
           'sorted_resp': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][9],
           'sorted_respMean': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][10],
           'ch2xy': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][16]}
            SET=Cebus2_M1_200123
        if m_i==2:    
            Macaque1_M1_181212 = scipy.io.loadmat(path_to_dataset+'/Macaque1_M1_181212.mat')
            Macaque1_M1_181212= {'emgs': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][0][0],
           'nChan': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][2][0][0],
           'sorted_isvalid': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][8],
           'sorted_resp': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][9],              
           'sorted_respMean': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][15],
           'ch2xy': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][14]}            
            SET=Macaque1_M1_181212
        if m_i==3:    
            Macaque2_M1_190527 = scipy.io.loadmat(path_to_dataset+'/Macaque2_M1_190527.mat')
            Macaque2_M1_190527= {'emgs': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][0][0],
           'nChan': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][2][0][0],
           'sorted_isvalid': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][8],
           'sorted_resp': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][9],              
           'sorted_respMean': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][15],
           'ch2xy': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][14]}
            SET=Macaque2_M1_190527
    elif dataset=='rat':   
        if m_i==0:
            rat1_M1_190716 = scipy.io.loadmat(path_to_dataset+'/rat1_M1_190716.mat')
            rat1_M1_190716= {'emgs': rat1_M1_190716['rat1_M1_190716'][0][0][0][0],
           'nChan': rat1_M1_190716['rat1_M1_190716'][0][0][2][0][0],
           'sorted_isvalid': rat1_M1_190716['rat1_M1_190716'][0][0][8],
           'sorted_resp': rat1_M1_190716['rat1_M1_190716'][0][0][9],              
           'sorted_respMean': rat1_M1_190716['rat1_M1_190716'][0][0][15],
           'ch2xy': rat1_M1_190716['rat1_M1_190716'][0][0][14]}            
            SET=rat1_M1_190716
        if m_i==1:
            rat2_M1_190617 = scipy.io.loadmat(path_to_dataset+'/rat2_M1_190617.mat')
            rat2_M1_190617= {'emgs': rat2_M1_190617['rat2_M1_190617'][0][0][0][0],
           'nChan': rat2_M1_190617['rat2_M1_190617'][0][0][2][0][0],
           'sorted_isvalid': rat2_M1_190617['rat2_M1_190617'][0][0][8],
           'sorted_resp': rat2_M1_190617['rat2_M1_190617'][0][0][9],              
           'sorted_respMean': rat2_M1_190617['rat2_M1_190617'][0][0][15],
           'ch2xy': rat2_M1_190617['rat2_M1_190617'][0][0][14]}         
            SET=rat2_M1_190617          
        if m_i==2:
            rat3_M1_190728 = scipy.io.loadmat(path_to_dataset+'/rat3_M1_190728.mat')
            rat3_M1_190728= {'emgs': rat3_M1_190728['rat3_M1_190728'][0][0][0][0],
           'nChan': rat3_M1_190728['rat3_M1_190728'][0][0][2][0][0],
           'sorted_isvalid': rat3_M1_190728['rat3_M1_190728'][0][0][8],
           'sorted_resp': rat3_M1_190728['rat3_M1_190728'][0][0][9],              
           'sorted_respMean': rat3_M1_190728['rat3_M1_190728'][0][0][15],
           'ch2xy': rat3_M1_190728['rat3_M1_190728'][0][0][14]}           
            SET=rat3_M1_190728                       
        if m_i==3:
            rat4_M1_191109 = scipy.io.loadmat(path_to_dataset+'/rat4_M1_191109.mat')
            rat4_M1_191109= {'emgs': rat4_M1_191109['rat4_M1_191109'][0][0][0][0],
           'nChan': rat4_M1_191109['rat4_M1_191109'][0][0][2][0][0],
           'sorted_isvalid': rat4_M1_191109['rat4_M1_191109'][0][0][8],
           'sorted_resp': rat4_M1_191109['rat4_M1_191109'][0][0][9],              
           'sorted_respMean': rat4_M1_191109['rat4_M1_191109'][0][0][15],
           'ch2xy': rat4_M1_191109['rat4_M1_191109'][0][0][14]}            
            SET=rat4_M1_191109                       
        if m_i==4:
            rat5_M1_191112 = scipy.io.loadmat(path_to_dataset+'/rat5_M1_191112.mat')
            rat5_M1_191112= {'emgs': rat5_M1_191112['rat5_M1_191112'][0][0][0][0],
           'nChan': rat5_M1_191112['rat5_M1_191112'][0][0][2][0][0],
           'sorted_isvalid': rat5_M1_191112['rat5_M1_191112'][0][0][8],
           'sorted_resp': rat5_M1_191112['rat5_M1_191112'][0][0][9],              
           'sorted_respMean': rat5_M1_191112['rat5_M1_191112'][0][0][15],
           'ch2xy': rat5_M1_191112['rat5_M1_191112'][0][0][14]}           
            SET=rat5_M1_191112                      
        if m_i==5:
            rat6_M1_200218 = scipy.io.loadmat(path_to_dataset+'/rat6_M1_200218.mat')        
            rat6_M1_200218= {'emgs': rat6_M1_200218['rat6_M1_200218'][0][0][0][0],
           'nChan': rat6_M1_200218['rat6_M1_200218'][0][0][2][0][0],
           'sorted_isvalid': rat6_M1_200218['rat6_M1_200218'][0][0][8],
           'sorted_resp': rat6_M1_200218['rat6_M1_200218'][0][0][9],              
           'sorted_respMean': rat6_M1_200218['rat6_M1_200218'][0][0][15],
           'ch2xy': rat6_M1_200218['rat6_M1_200218'][0][0][14]}          
            SET=rat6_M1_200218               
    else:
        print('Invalid value for dataset variable. Has to be either \'nhp\' or \'rat\'. ')     
        SET=None        
    return SET
