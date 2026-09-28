from guard import (
    is_ikram_question,
    get_rejection_message
)


def test_question(question, history=None):

    allowed = is_ikram_question(
        question,
        history
    )

    print("\nQuestion:", question)

    if allowed:

        print("STATUS: ALLOWED")

    else:

        print("STATUS: REJECTED")
        print("Response:", get_rejection_message())


print("=" * 60)
print("TEST 1")
print("=" * 60)

test_question(
    "Who is Ikram Ullah?"
)


print("=" * 60)
print("TEST 2")
print("=" * 60)

test_question(
    "What projects has Ikram completed?"
)


print("=" * 60)
print("TEST 3")
print("=" * 60)

test_question(
    "Who is Elon Musk?"
)


print("=" * 60)
print("TEST 4")
print("=" * 60)

test_question(
    "What is machine learning?"
)


print("=" * 60)
print("TEST 5")
print("=" * 60)

history = [
    "Who is Ikram Ullah?",
    "Ikram Ullah is a BS Artificial Intelligence student."
]

test_question(
    "What projects has he completed?",
    history
)