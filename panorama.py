def stitch(inp_path, imgmark, N=4, savepath=''): 
    imgpath = [f'{inp_path}/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)

    overlap_arr = []

    sift = cv2.SIFT_create()
    
    bf = cv2.BFMatcher()
    for i in range(1, len(imgs)):
        grayImg1 = cv2.cvtColor(imgs[i-1],cv2.COLOR_BGR2GRAY)
        grayImg2 = cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY)
        
        width = grayImg2.shape[1] + grayImg1.shape[1]
        height = max(grayImg2.shape[0], grayImg1.shape[0])

        kps1, des1 = sift.detectAndCompute(grayImg2,None)
        kps2, des2 = sift.detectAndCompute(grayImg1,None)
        matche = bf.knnMatch(des1, des2, k=2)

        goodMatch = []
        for m,n in matche:
            if m.distance < 0.8 * n.distance:
                goodMatch.append(m)

        sourcePts = []
        destPts = []

        for m in goodMatch:
            sourcePts = np.float32([ kps1[m.queryIdx].pt for m in goodMatch ])
            sourcePts = sourcePts.reshape(-1,1,2)
            destPts = np.float32([ kps2[m.trainIdx].pt for m in goodMatch ])
            destPts = destPts.reshape(-1,1,2)
        matrix, mask = cv2.findHomography(sourcePts, destPts, cv2.RANSAC, 5.0)

        dest = cv2.warpPerspective(imgs[i],matrix,(width, height))
        dest[0:imgs[i-1].shape[0], 0:imgs[i-1].shape[1]] = imgs[i-1]
        def trimBlank(imgFit):
            if not np.sum(imgFit[0]):
                return trimBlank(imgFit[1:])
            if not np.sum(imgFit[:,-1]):
                return trimBlank(imgFit[:,:-2])  
            if not np.sum(imgFit[:,0]):
                return trimBlank(imgFit[:,1:])                             
            if not np.sum(imgFit[-1]):
                return trimBlank(imgFit[:-2])

            return imgFit

        imgs[i] = trimBlank(dest)
        imgs[i] = dest

    cv2.imwrite("task2_result.png", trimBlank(dest))
    return ""
