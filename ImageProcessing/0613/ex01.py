import cv2

img1= cv2.imread('../data/apple.png',cv2.IMREAD_GRAYSCALE)
img2= cv2.imread('../data/apple.png',cv2.IMREAD_GRAYSCALE)

# 특징점 검출기 생성
orb=cv2.ORB_create()

# 특징점 검출과 디스크럽터 계산
keypoint01,descriptor01= orb.detectAndCompute(img1,None)
keypoint02,descriptor02= orb.detectAndCompute(img2,None)

# 매칭기 생성
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

# 특징점 매칭
matches=bf.match(descriptor01,descriptor02)

# 매칭 결과 정렬
matches=sorted(matches,key=lambda x:x.distance)