import glob
from shutil import copyfile
emotions = ["neutral","anger","contempt","disgust","fear","happy","sadness","surprise"]
participants = glob.glob("./source_emotion/*")
for x in participants:
    part = "%s" % x[-4:]
    for sessions in glob.glob("%s/*" %x):
        for files in glob.glob("%s/*" %sessions):
            current_session = files[22:-30]
            file=open(files,'r')
            emotion = int(float(file.readline()))

            image_files = glob.glob("source_images/%s/%s/*" %(part,current_session))
            sourcefile_emotion = image_files[-1]
            sourcefile_neutral = image_files[0]

            dest_neut = "./sorted_set/neutral/%s" %sourcefile_neutral[23:]
            dest_emot = "./sorted_set/%s/%s" %(emotions[emotion],sourcefile_emotion[23:])

            copyfile(sourcefile_neutral,dest_neut)
            copyfile(sourcefile_emotion,dest_emot)