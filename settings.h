#ifndef SETTINGS_H
#define SETTINGS_H

#define	TRANSLATION_DECAY			1.0
#define ROTATION_DECAY			1.0

#define JELLO_DECAY 				0.95

#define SHOW_CORNERS            		0

#define SVD_PRUNE_MAX_DIST      		2.

#define WIN_SIZE				21
#define NUM_CORNERS			1000

#define SVD_WEIGHT_FUNC3(d) 		(exp(-pow((d)/40., 2)))
#define SVD_ROWS    				41

#define DO_CORNER_SUBPIX        		1

#define	 NUM_FRAMES				100

#define REMOVE_GAUSSIAN_WEIGHT_TAILS	0

#if(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/GenerateTestVideo/rotateFuf.avi"
#define OUTPUT_FILENAME		"fuf-rotated.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/GenerateTestVideo2/testVideo1.avi"
#define	SHFITS_FILENAME		"../GenerateTestVideo2/shifts/shakeRot%d.txt"
#define OUTPUT_FILENAME		"testVideo1.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Desktop/jelloVideos/trails.mp4"
#define OUTPUT_FILENAME		"/Users/nickstupich/Desktop/v17_output/trails.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Desktop/jelloVideos/jelloTest720.mov"
#define OUTPUT_FILENAME		"/Users/nickstupich/Desktop/v16_output/parliament720.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Desktop/jelloVideos/jelloTest.mov"
#define OUTPUT_FILENAME		"/Users/nickstupich/Desktop/v16_output/parliament1080.avi"

#elif(0)
#define INPUT_FILENAME		"parliament4.avi"
#define OUTPUT_FILENAME		"parliament5.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/nexus4_1.mp4"
#define OUTPUT_FILENAME		"nexus4_1.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/nexus4_3.mp4"
#define OUTPUT_FILENAME		"nexus4_3.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/17362720.mp4"
#define OUTPUT_FILENAME		"17362720.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/race720.mov"
#define OUTPUT_FILENAME		"race.avi"

#elif(1)
#define INPUT_FILENAME		"/Users/nickstupich/Desktop/jelloVideos/driving720.mov"
#define OUTPUT_FILENAME		"/Users/nickstupich/Desktop/v16_output/driving.avi"

#elif(1)
#define INPUT_FILENAME		"/Users/nickstupich/Desktop/jelloVideos/walking1080.mp4"
#define OUTPUT_FILENAME		"/Users/nickstupich/Desktop/v16_output/walking1080.avi"
#define ROTATE90


#elif(0)
#define INPUT_FILENAME		"walking1080Sideways.avi"
#define OUTPUT_FILENAME		"walking1080_2.avi"
#define ROTATE90

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/poster_landscape.mp4"
#define OUTPUT_FILENAME		"poster_landscape.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/hip_whip.mov"
#define OUTPUT_FILENAME		"hip_whip.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/bike_grass.mp4"
#define OUTPUT_FILENAME		"bike_grass.avi"

#elif(0)
#define INPUT_FILENAME		"bike_grass.avi"
#define OUTPUT_FILENAME		"bike_grass2.avi"


#elif(1)
#define INPUT_FILENAME		"/Users/nickstupich/Desktop/jelloVideos/bike_street.mp4"
#define OUTPUT_FILENAME		"/Users/nickstupich/Desktop/v17_output/bike_street.avi"


#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/3ToSmith.mp4"
#define OUTPUT_FILENAME		"3ToSmith.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/dog1.mp4"
#define OUTPUT_FILENAME		"dog1.avi"

#elif(0)
#define INPUT_FILENAME		"/Users/nickstupich/Dropbox/grad/other/Jello2/fliu_3.avi"
#define OUTPUT_FILENAME		"fliu_3.avi"

#elif(1)
#define INPUT_FILENAME 		"/Users/nickstupich/Desktop/jelloVideos/laptop.mp4"
#define OUTPUT_FILENAME		"/Users/nickstupich/Desktop/v16_output/laptop.avi"


#endif





#endif
