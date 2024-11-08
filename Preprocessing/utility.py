
def get_z(dicom,x,y):
    H,W = dicom.pixel_array.shape
    sx,sy,sz = [float(v) for v in dicom.ImagePositionPatient]
    o0, o1, o2, o3, o4, o5, = [float(v) for v in dicom.ImageOrientationPatient]
    delx,dely = dicom.PixelSpacing
    xx =  o0*delx*x + o3*dely*y + sx
    yy =  o1*delx*x + o4*dely*y + sy
    zz =  o2*delx*x + o5*dely*y + sz
    return H,W,xx,yy,zz