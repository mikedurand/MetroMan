#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import concatenate, zeros, tril,ones

def CalcU(D):
        
    M=D.nR * D.nt
    N=D.nR *(D.nt-1)
    
    u=concatenate( (zeros( (1,D.nt-1) ), tril(ones( (D.nt-1,D.nt-1) )) ),0  )
    
    U=zeros( (M,N) )
    
    for i in range(0,D.nR):
        a=D.nt*i
        b=D.nt*i+D.nt
        c=(D.nt-1)*i
        d=(D.nt-1)*i+(D.nt-1)
        U[a:b,c:d  ] = u
        
    return U

