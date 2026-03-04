from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))
print("Model downloaded.")