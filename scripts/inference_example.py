from echogemma import EchoGemma

eg = EchoGemma()
stack_of_videos = eg.process_dicoms("example_study")
report=eg.generate(stack_of_videos)
print(report)