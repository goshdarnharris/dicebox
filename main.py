import ImageSource

# Our local vision library. Maybe change name?
import vision

try:
    i = 0
    while True:
        print("Grabbing image...")
        corrected_img = ImageSource.getImage()
        print("Starting recognition...")
        results = vision.do_recognition(corrected_img, "livecam")
        print(results)
        roll_counts = [results.count(val) for val in range(1,7)]
        print(roll_counts)
        print("Counted %d dice"%len(results))
        #os.system("eog -f livecam/overlay.png")
except KeyboardInterrupt:
    ImageSource.close()
    print("Goodbye!")


