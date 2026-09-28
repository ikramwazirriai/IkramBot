from rag import IkrambotRAG


def main():

    print("=" * 60)
    print("IKRAMBOT RAG TEST")
    print("=" * 60)

    chatbot = IkrambotRAG()

    history = []

    while True:

        question = input(
            "\nYou: "
        )

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print("Goodbye!")

            break

        result = chatbot.ask(
            question,
            history=history
        )

        print("\nIkrambot:")
        print(result["answer"])

        if result["sources"]:

            print("\nSources:")

            for source in result["sources"]:

                print(f"- {source}")

        # --------------------------------------
        # Save conversation
        # --------------------------------------

        history.append(
            {
                "role": "user",
                "content": question
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": result["answer"]
            }
        )


if __name__ == "__main__":
    main()