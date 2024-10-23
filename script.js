document.getElementById('summarizeBtn').addEventListener('click', async () => {
    const inputText = document.getElementById('inputText').value;
    const summaryOption = document.getElementById('summaryOption').value;
    const numSentences = document.getElementById('numSentences').value;

    console.log("Button clicked!");
    console.log("Input Text:", inputText);
    console.log("Summary Option:", summaryOption);
    console.log("Number of Sentences:", numSentences);

    try {
        const response = await fetch('http://127.0.0.1:5000/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: inputText,
                num_sentences: numSentences,
                technique: summaryOption,
            }),
        });

        console.log("Response received:", response);
        const data = await response.json();
        const summaryOutput = document.getElementById('summaryOutput');

        if (response.ok) {
            summaryOutput.textContent = data.summary;
        } else {
            summaryOutput.textContent = `Error: ${data.error}`;
        }
    } catch (error) {
        console.error("Error during fetch:", error);
        document.getElementById('summaryOutput').textContent = "An error occurred during summarization.";
    }
});
