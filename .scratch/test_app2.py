from streamlit.testing.v1 import AppTest

at = AppTest.from_file("../app.py", default_timeout=120)
at.run()
at.chat_input[0].set_value("Suggest some chill songs for studying.").run()

print("exceptions:", at.exception)
print()
# Inspect the full tree of the last chat_message (assistant response)
last = at.chat_message[-1]
print("last message role:", last.type)
for md in last.markdown:
    print("MARKDOWN:", md.value[:300])
for cap in last.caption:
    print("CAPTION:", cap.value)

print("\n--- full app markdown dump (first 5) ---")
for md in at.markdown[:5]:
    print(repr(md.value[:150]))

print("\n--- unknown song test ---")
at2 = AppTest.from_file("../app.py", default_timeout=120)
at2.run()
at2.chat_input[0].set_value("Recommend songs similar to XYZ123FakeSong").run()
print("exceptions:", at2.exception)
last2 = at2.chat_message[-1]
for md in last2.markdown:
    print("MARKDOWN:", md.value)
