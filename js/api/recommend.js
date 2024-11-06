const express = require("express");
const cors = require("cors");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const csv = require("csv-parser");
const tf = require('@tensorflow/tfjs-node');

const MODEL_URL = "https://drive.google.com/uc?export=download&id=1WKIbyp-dqRVNE7602teHVHnKzQ0Elo9m";
const DATASET_URL = "https://drive.google.com/uc?export=download&id=1-8CQ7lqRJk-VMkFy4h8nUE0X-C2VFazq";

const MODEL_FILE = "model.json";
const DATASET_FILE = "SpotifyFeatures.csv";

const app = express();
app.use(cors());
app.use(express.json());

let data = [];
let model;

async function downloadFile(url, filename) {
    const filePath = path.join(__dirname, filename);
    if (!fs.existsSync(filePath)) {
        console.log(`Mengunduh ${filename}...`);
        const response = await axios.get(url, { responseType: "stream" });
        response.data.pipe(fs.createWriteStream(filePath));
        await new Promise(resolve => response.data.on("end", resolve));
        console.log(`${filename} berhasil diunduh.`);
    }
}

async function loadResources() {
    await downloadFile(MODEL_URL, MODEL_FILE);
    await downloadFile(DATASET_URL, DATASET_FILE);

    fs.createReadStream(DATASET_FILE)
        .pipe(csv())
        .on("data", row => data.push(row))
        .on("end", () => console.log("Dataset berhasil dimuat"));

    model = await tf.loadLayersModel(`file://${path.join(__dirname, MODEL_FILE)}`);
    console.log("Model berhasil dimuat.");
}

app.post("/recommend", async (req, res) => {
    if (!req.body.mood) {
        return res.status(400).json({ error: "Mood parameter is required" });
    }
    if (!model) {
        return res.status(500).json({ error: "Model not available" });
    }

    const userInput = req.body.mood;
    const userInputTensor = tf.tensor([userInput]);

    const predictions = model.predict(userInputTensor);
    const recommendations = predictions.arraySync()[0].map(index => data[index]);

    res.json(recommendations);
});

loadResources();

module.exports = app;
