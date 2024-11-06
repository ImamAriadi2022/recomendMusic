const express = require('express');
const cors = require('cors');
const axios = require('axios');
const fs = require('fs');
const csv = require('csv-parser');
const KNN = require('ml-knn');
const path = require('path');

// URL Google Drive untuk model dan dataset
const MODEL_URL = "https://drive.google.com/uc?export=download&id=1WKIbyp-dqRVNE7602teHVHnKzQ0Elo9m";
const DATASET_URL = "https://drive.google.com/uc?export=download&id=1-8CQ7lqRJk-VMkFy4h8nUE0X-C2VFazq";

// Nama file lokal untuk model dan dataset
const MODEL_FILE = path.join(__dirname, "model.json");
const DATASET_FILE = path.join(__dirname, "SpotifyFeatures.csv");

const app = express();
app.use(cors());
app.use(express.json());

let knnModel;
let data = [];

// Fungsi untuk mengunduh file dari URL jika belum ada di lokal
const downloadFile = async (url, filePath) => {
    if (!fs.existsSync(filePath)) {
        console.log(`Mengunduh ${filePath}...`);
        const response = await axios({ url, method: 'GET', responseType: 'stream' });
        const writer = fs.createWriteStream(filePath);
        response.data.pipe(writer);
        return new Promise((resolve, reject) => {
            writer.on('finish', resolve);
            writer.on('error', reject);
        });
    } else {
        console.log(`${filePath} sudah ada, melewati unduhan.`);
    }
};

// Fungsi untuk memuat dataset
const loadDataset = () => {
    return new Promise((resolve, reject) => {
        const records = [];
        fs.createReadStream(DATASET_FILE)
            .pipe(csv())
            .on('data', (row) => records.push(row))
            .on('end', () => {
                console.log("Dataset berhasil dimuat.");
                data = records;
                resolve();
            })
            .on('error', reject);
    });
};


const loadModel = () => {
    if (fs.existsSync('Model.json')){
        console.log("memuattt modelll.....");
        const modelData = JSON.parse(fs.readFileSync('Model.json'));

        // model kkknnnn
        const kknModel = {
            n_neighbors: modelData.n_neighbors,
            trainingSet: modelData.trainingSet
        };

        console.log("model berhasil dimuat...");
        return kknModel;
    } else {
        console.error('model tidak ditemulam')
    }
};


// Endpoint rekomendasi
app.post('/recommend', (req, res) => {
    try {
        const mood = req.body.mood;
        if (!mood) {
            return res.status(400).json({ error: "Parameter 'mood' diperlukan" });
        }

        if (!knnModel) {
            return res.status(500).json({ error: "Model tidak tersedia" });
        }

        const [nearestIndexes] = knnModel.predict([mood]);
        const recommendedSongs = nearestIndexes.map(index => data[index]);

        res.json(recommendedSongs);
    } catch (error) {
        console.error("Terjadi kesalahan:", error);
        res.status(500).json({ error: "Kesalahan Internal Server" });
    }
});

// Fungsi utama untuk menyiapkan server
const setupServer = async () => {
    try {
        await downloadFile(MODEL_URL, MODEL_FILE);
        await downloadFile(DATASET_URL, DATASET_FILE);
        await loadDataset();
        loadModel();

        app.listen(4000, () => {
            console.log("Server berjalan di http://localhost:4000");
        });
    } catch (error) {
        console.error("Gagal menyiapkan server:", error);
    }
};

setupServer();
