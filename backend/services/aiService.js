const tf = require('@tensorflow/tfjs-node');
const axios = require('axios');
const winston = require('winston');

class AIService {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.models = {};
        this.logger = winston.createLogger({
            level: 'info',
            format: winston.format.simple(),
            transports: [new winston.transports.Console()]
        });
        
        this.initializeModels();
    }

    async initializeModels() {
        try {
            this.logger.info('Initializing AI models...');
            
            // Initialize disease detection CNN model
            this.models.diseaseDetection = await this.createDiseaseModel();
            
            // Initialize crop stress LSTM model
            this.models.stressDetection = await this.createStressModel();
            
            // Initialize yield prediction model
            this.models.yieldPrediction = await this.createYieldModel();
            
            this.logger.info('AI models initialized successfully');
        } catch (error) {
            this.logger.error('Failed to initialize AI models:', error);
            throw error;
        }
    }

    // Disease Detection Model (CNN)
    async createDiseaseModel() {
        // Create a simplified CNN model for disease detection
        const model = tf.sequential({
            layers: [
                tf.layers.conv2d({
                    inputShape: [224, 224, 3],
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),
                tf.layers.conv2d({
                    filters: 64,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),
                tf.layers.conv2d({
                    filters: 128,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),
                tf.layers.flatten(),
                tf.layers.dense({ units: 512, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.5 }),
                tf.layers.dense({ units: 10, activation: 'softmax' }) // 10 disease classes
            ]
        });

        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    // Stress Detection Model (LSTM)
    async createStressModel() {
        const model = tf.sequential({
            layers: [
                tf.layers.lstm({
                    units: 50,
                    returnSequences: true,
                    inputShape: [30, 5] // 30 time steps, 5 features
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.lstm({
                    units: 50,
                    returnSequences: false
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 25, activation: 'relu' }),
                tf.layers.dense({ units: 3, activation: 'softmax' }) // Low, Medium, High stress
            ]
        });

        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    // Yield Prediction Model
    async createYieldModel() {
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    units: 128,
                    activation: 'relu',
                    inputShape: [15] // 15 input features
                }),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.dense({ units: 64, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.dense({ units: 32, activation: 'relu' }),
                tf.layers.dense({ units: 1, activation: 'linear' }) // Regression output
            ]
        });

        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    // Hyperspectral Image Analysis
    async calculateVegetationIndices(hyperspectralData) {
        try {
            const { bands, metadata } = hyperspectralData;
            
            // Extract key bands for vegetation indices
            const red = bands.find(b => b.wavelength >= 650 && b.wavelength <= 680);
            const nir = bands.find(b => b.wavelength >= 780 && b.wavelength <= 900);
            const redEdge = bands.find(b => b.wavelength >= 700 && b.wavelength <= 750);
            const blue = bands.find(b => b.wavelength >= 450 && b.wavelength <= 520);
            const green = bands.find(b => b.wavelength >= 520 && b.wavelength <= 600);

            if (!red || !nir) {
                throw new Error('Required bands not found in hyperspectral data');
            }

            // Calculate vegetation indices
            const ndvi = this.calculateNDVI(red.data, nir.data);
            const evi = this.calculateEVI(red.data, nir.data, blue?.data);
            const savi = this.calculateSAVI(red.data, nir.data);
            const ndre = this.calculateNDRE(redEdge?.data, nir.data);
            const gndvi = this.calculateGNDVI(green?.data, nir.data);

            // Use Google Cloud Vision API for enhanced analysis
            const enhancedAnalysis = await this.enhanceWithGoogleVision(hyperspectralData);

            return {
                ndvi,
                evi,
                savi,
                ndre,
                gndvi,
                enhanced: enhancedAnalysis,
                confidence: this.calculateConfidence([ndvi, evi, savi, ndre, gndvi]),
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            this.logger.error('Vegetation indices calculation failed:', error);
            throw error;
        }
    }

    calculateNDVI(red, nir) {
        // NDVI = (NIR - Red) / (NIR + Red)
        const redMean = this.calculateMean(red);
        const nirMean = this.calculateMean(nir);
        
        if (nirMean + redMean === 0) return 0;
        return (nirMean - redMean) / (nirMean + redMean);
    }

    calculateEVI(red, nir, blue) {
        // EVI = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
        const redMean = this.calculateMean(red);
        const nirMean = this.calculateMean(nir);
        const blueMean = blue ? this.calculateMean(blue) : 0;
        
        const denominator = nirMean + 6 * redMean - 7.5 * blueMean + 1;
        if (denominator === 0) return 0;
        
        return 2.5 * ((nirMean - redMean) / denominator);
    }

    calculateSAVI(red, nir, L = 0.5) {
        // SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        const redMean = this.calculateMean(red);
        const nirMean = this.calculateMean(nir);
        
        const denominator = nirMean + redMean + L;
        if (denominator === 0) return 0;
        
        return ((nirMean - redMean) / denominator) * (1 + L);
    }

    calculateNDRE(redEdge, nir) {
        if (!redEdge || !nir) return null;
        
        // NDRE = (NIR - RedEdge) / (NIR + RedEdge)
        const redEdgeMean = this.calculateMean(redEdge);
        const nirMean = this.calculateMean(nir);
        
        if (nirMean + redEdgeMean === 0) return 0;
        return (nirMean - redEdgeMean) / (nirMean + redEdgeMean);
    }

    calculateGNDVI(green, nir) {
        if (!green || !nir) return null;
        
        // GNDVI = (NIR - Green) / (NIR + Green)
        const greenMean = this.calculateMean(green);
        const nirMean = this.calculateMean(nir);
        
        if (nirMean + greenMean === 0) return 0;
        return (nirMean - greenMean) / (nirMean + greenMean);
    }

    calculateMean(array) {
        if (!array || array.length === 0) return 0;
        return array.reduce((sum, val) => sum + val, 0) / array.length;
    }

    calculateConfidence(indices) {
        // Calculate confidence based on the consistency of vegetation indices
        const validIndices = indices.filter(idx => idx !== null && !isNaN(idx));
        if (validIndices.length === 0) return 0;
        
        const mean = validIndices.reduce((sum, idx) => sum + idx, 0) / validIndices.length;
        const variance = validIndices.reduce((sum, idx) => sum + Math.pow(idx - mean, 2), 0) / validIndices.length;
        
        // Higher consistency (lower variance) = higher confidence
        return Math.max(0, 1 - variance * 2);
    }

    // Google Cloud Vision API Integration
    async enhanceWithGoogleVision(hyperspectralData) {
        try {
            if (!this.apiKey) {
                this.logger.warn('Google API key not available, skipping enhanced analysis');
                return null;
            }

            // Convert hyperspectral data to RGB for Google Vision API
            const rgbImage = this.convertToRGB(hyperspectralData);
            
            const response = await axios.post(
                `https://vision.googleapis.com/v1/images:annotate?key=${this.apiKey}`,
                {
                    requests: [{
                        image: {
                            content: rgbImage
                        },
                        features: [
                            { type: 'LABEL_DETECTION', maxResults: 10 },
                            { type: 'IMAGE_PROPERTIES', maxResults: 10 },
                            { type: 'OBJECT_LOCALIZATION', maxResults: 10 }
                        ]
                    }]
                }
            );

            return this.processGoogleVisionResponse(response.data);
        } catch (error) {
            this.logger.error('Google Vision API error:', error.message);
            return null;
        }
    }

    convertToRGB(hyperspectralData) {
        // Convert hyperspectral bands to RGB for Google Vision API
        // This is a simplified conversion - in production, you'd use more sophisticated methods
        const { bands } = hyperspectralData;
        
        const red = bands.find(b => b.wavelength >= 630 && b.wavelength <= 700);
        const green = bands.find(b => b.wavelength >= 520 && b.wavelength <= 600);
        const blue = bands.find(b => b.wavelength >= 450 && b.wavelength <= 520);
        
        // Create RGB image buffer (simplified)
        const width = 224, height = 224;
        const rgbBuffer = Buffer.alloc(width * height * 3);
        
        for (let i = 0; i < width * height; i++) {
            rgbBuffer[i * 3] = red ? Math.min(255, red.data[i] * 255) : 0;
            rgbBuffer[i * 3 + 1] = green ? Math.min(255, green.data[i] * 255) : 0;
            rgbBuffer[i * 3 + 2] = blue ? Math.min(255, blue.data[i] * 255) : 0;
        }
        
        return rgbBuffer.toString('base64');
    }

    processGoogleVisionResponse(response) {
        if (!response.responses || !response.responses[0]) {
            return null;
        }

        const result = response.responses[0];
        
        return {
            labels: result.labelAnnotations?.map(label => ({
                description: label.description,
                score: label.score,
                confidence: label.score
            })) || [],
            
            objects: result.localizedObjectAnnotations?.map(obj => ({
                name: obj.name,
                confidence: obj.score,
                boundingBox: obj.boundingPoly
            })) || [],
            
            dominantColors: result.imagePropertiesAnnotation?.dominantColors?.colors?.map(color => ({
                color: color.color,
                score: color.score,
                pixelFraction: color.pixelFraction
            })) || [],
            
            cropHealth: this.analyzeCropHealthFromLabels(result.labelAnnotations || [])
        };
    }

    analyzeCropHealthFromLabels(labels) {
        const healthKeywords = {
            healthy: ['green', 'vegetation', 'leaf', 'plant', 'crop', 'agriculture'],
            diseased: ['yellow', 'brown', 'wilted', 'diseased', 'fungus', 'pest'],
            stressed: ['dry', 'drought', 'stress', 'pale', 'deficiency']
        };

        let healthScore = {
            healthy: 0,
            diseased: 0,
            stressed: 0
        };

        labels.forEach(label => {
            const description = label.description.toLowerCase();
            const score = label.score;

            Object.keys(healthKeywords).forEach(category => {
                if (healthKeywords[category].some(keyword => description.includes(keyword))) {
                    healthScore[category] += score;
                }
            });
        });

        const totalScore = Object.values(healthScore).reduce((sum, score) => sum + score, 0);
        if (totalScore === 0) return null;

        // Normalize scores
        Object.keys(healthScore).forEach(key => {
            healthScore[key] = healthScore[key] / totalScore;
        });

        return healthScore;
    }

    // Spectral Anomaly Detection
    async detectSpectralAnomalies(hyperspectralData) {
        try {
            const { bands, metadata } = hyperspectralData;
            const anomalies = [];

            // Statistical anomaly detection
            const statisticalAnomalies = this.detectStatisticalAnomalies(bands);
            anomalies.push(...statisticalAnomalies);

            // Spectral signature anomalies
            const spectralAnomalies = this.detectSpectralSignatureAnomalies(bands);
            anomalies.push(...spectralAnomalies);

            // Machine learning based anomaly detection
            const mlAnomalies = await this.detectMLAnomalies(bands);
            anomalies.push(...mlAnomalies);

            return {
                anomalies,
                summary: {
                    totalAnomalies: anomalies.length,
                    severityDistribution: this.calculateSeverityDistribution(anomalies),
                    confidence: this.calculateAnomalyConfidence(anomalies)
                }
            };
        } catch (error) {
            this.logger.error('Anomaly detection failed:', error);
            throw error;
        }
    }

    detectStatisticalAnomalies(bands) {
        const anomalies = [];
        
        bands.forEach((band, index) => {
            const mean = this.calculateMean(band.data);
            const std = this.calculateStandardDeviation(band.data, mean);
            
            // Z-score based anomaly detection
            band.data.forEach((value, pixelIndex) => {
                const zScore = Math.abs((value - mean) / std);
                
                if (zScore > 3) { // 3-sigma rule
                    anomalies.push({
                        type: 'statistical',
                        wavelength: band.wavelength,
                        pixelIndex,
                        severity: zScore > 4 ? 'high' : 'medium',
                        confidence: Math.min(0.95, zScore / 4),
                        description: `Statistical anomaly at ${band.wavelength}nm (Z-score: ${zScore.toFixed(2)})`
                    });
                }
            });
        });

        return anomalies.slice(0, 100); // Limit to top 100 anomalies
    }

    detectSpectralSignatureAnomalies(bands) {
        const anomalies = [];
        
        // Define expected vegetation spectral signature patterns
        const expectedPatterns = this.getExpectedVegetationPatterns();
        
        // Compare actual spectra with expected patterns
        const pixelCount = bands[0].data.length;
        
        for (let pixelIndex = 0; pixelIndex < Math.min(pixelCount, 1000); pixelIndex += 10) {
            const pixelSpectrum = bands.map(band => band.data[pixelIndex]);
            
            // Calculate deviation from expected patterns
            const deviation = this.calculateSpectralDeviation(pixelSpectrum, expectedPatterns);
            
            if (deviation.score > 0.7) {
                anomalies.push({
                    type: 'spectral_signature',
                    pixelIndex,
                    severity: deviation.score > 0.85 ? 'high' : 'medium',
                    confidence: deviation.score,
                    description: `Spectral signature anomaly: ${deviation.reason}`,
                    spectrum: pixelSpectrum
                });
            }
        }

        return anomalies;
    }

    async detectMLAnomalies(bands) {
        // Simplified ML-based anomaly detection
        // In production, this would use pre-trained models
        
        const anomalies = [];
        
        try {
            // Create feature matrix from spectral bands
            const features = this.extractSpectralFeatures(bands);
            
            // Simulate ML model prediction
            const predictions = await this.predictAnomalies(features);
            
            predictions.forEach((prediction, index) => {
                if (prediction.isAnomaly) {
                    anomalies.push({
                        type: 'ml_detected',
                        pixelIndex: index * 10, // Sampled pixels
                        severity: prediction.confidence > 0.8 ? 'high' : 'medium',
                        confidence: prediction.confidence,
                        description: `ML-detected anomaly: ${prediction.anomalyType}`,
                        features: prediction.features
                    });
                }
            });
        } catch (error) {
            this.logger.error('ML anomaly detection failed:', error);
        }

        return anomalies;
    }

    calculateStandardDeviation(array, mean) {
        const variance = array.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / array.length;
        return Math.sqrt(variance);
    }

    getExpectedVegetationPatterns() {
        return {
            healthy: {
                visible: { min: 0.03, max: 0.12 }, // Low reflectance in visible
                nir: { min: 0.7, max: 0.9 }, // High reflectance in NIR
                redEdge: { slope: 'steep_positive' } // Steep increase at red edge
            },
            stressed: {
                visible: { min: 0.05, max: 0.18 },
                nir: { min: 0.4, max: 0.7 },
                redEdge: { slope: 'moderate_positive' }
            },
            diseased: {
                visible: { min: 0.08, max: 0.25 },
                nir: { min: 0.3, max: 0.6 },
                redEdge: { slope: 'flat' }
            }
        };
    }

    calculateSpectralDeviation(spectrum, patterns) {
        // Simplified spectral pattern matching
        const visibleRange = spectrum.slice(0, 40); // Approximate visible range
        const nirRange = spectrum.slice(60, 80); // Approximate NIR range
        
        const visibleMean = this.calculateMean(visibleRange);
        const nirMean = this.calculateMean(nirRange);
        
        let maxDeviation = 0;
        let reason = '';
        
        Object.keys(patterns).forEach(patternType => {
            const pattern = patterns[patternType];
            
            let deviation = 0;
            
            // Check visible range
            if (visibleMean < pattern.visible.min || visibleMean > pattern.visible.max) {
                deviation += 0.3;
            }
            
            // Check NIR range
            if (nirMean < pattern.nir.min || nirMean > pattern.nir.max) {
                deviation += 0.4;
            }
            
            // Check red edge slope (simplified)
            const redEdgeSlope = this.calculateRedEdgeSlope(spectrum);
            if (this.evaluateRedEdgeSlope(redEdgeSlope, pattern.redEdge.slope)) {
                deviation += 0.3;
            }
            
            if (deviation > maxDeviation) {
                maxDeviation = deviation;
                reason = `Deviation from ${patternType} vegetation pattern`;
            }
        });
        
        return { score: maxDeviation, reason };
    }

    calculateRedEdgeSlope(spectrum) {
        // Calculate slope in red edge region (approximately bands 40-60)
        const redEdgeStart = 40;
        const redEdgeEnd = 60;
        
        const startValue = spectrum[redEdgeStart];
        const endValue = spectrum[redEdgeEnd];
        
        return (endValue - startValue) / (redEdgeEnd - redEdgeStart);
    }

    evaluateRedEdgeSlope(slope, expectedSlope) {
        // Evaluate if slope matches expected pattern
        switch (expectedSlope) {
            case 'steep_positive': return slope < 0.01;
            case 'moderate_positive': return slope < 0.005;
            case 'flat': return Math.abs(slope) > 0.002;
            default: return false;
        }
    }

    extractSpectralFeatures(bands) {
        // Extract key features for ML analysis
        const features = [];
        const sampleCount = Math.min(1000, bands[0].data.length);
        
        for (let i = 0; i < sampleCount; i += 10) {
            const spectrum = bands.map(band => band.data[i]);
            
            const feature = {
                mean: this.calculateMean(spectrum),
                std: this.calculateStandardDeviation(spectrum, this.calculateMean(spectrum)),
                min: Math.min(...spectrum),
                max: Math.max(...spectrum),
                redEdgeSlope: this.calculateRedEdgeSlope(spectrum),
                nirPlateau: this.calculateMean(spectrum.slice(60, 80))
            };
            
            features.push(feature);
        }
        
        return features;
    }

    async predictAnomalies(features) {
        // Simulate ML model predictions
        return features.map(feature => {
            // Simple rule-based simulation of ML predictions
            let isAnomaly = false;
            let confidence = 0;
            let anomalyType = 'normal';
            
            // Simulate different types of anomalies
            if (feature.nirPlateau < 0.4) {
                isAnomaly = true;
                confidence = 0.8;
                anomalyType = 'stress_detected';
            } else if (feature.redEdgeSlope < 0.001) {
                isAnomaly = true;
                confidence = 0.75;
                anomalyType = 'disease_signature';
            } else if (feature.std > 0.3) {
                isAnomaly = true;
                confidence = 0.65;
                anomalyType = 'high_variability';
            }
            
            return {
                isAnomaly,
                confidence,
                anomalyType,
                features: feature
            };
        });
    }

    calculateSeverityDistribution(anomalies) {
        const distribution = { high: 0, medium: 0, low: 0 };
        
        anomalies.forEach(anomaly => {
            distribution[anomaly.severity] = (distribution[anomaly.severity] || 0) + 1;
        });
        
        return distribution;
    }

    calculateAnomalyConfidence(anomalies) {
        if (anomalies.length === 0) return 0;
        
        const totalConfidence = anomalies.reduce((sum, anomaly) => sum + anomaly.confidence, 0);
        return totalConfidence / anomalies.length;
    }

    // Disease Prediction
    async predictDisease(imageData, environmentalData) {
        try {
            // Preprocess image data for CNN model
            const processedImage = await this.preprocessImageForCNN(imageData);
            
            // Make prediction using the disease detection model
            const prediction = this.models.diseaseDetection.predict(processedImage);
            const probabilities = await prediction.data();
            
            // Get disease classes
            const diseaseClasses = this.getDiseaseClasses();
            const topPrediction = this.getTopPrediction(probabilities, diseaseClasses);
            
            // Incorporate environmental data
            const environmentalFactor = this.calculateEnvironmentalFactor(environmentalData);
            
            // Calculate final confidence
            const finalConfidence = topPrediction.confidence * environmentalFactor;
            
            return {
                disease: topPrediction.disease,
                confidence: finalConfidence,
                severity: this.calculateSeverity(finalConfidence),
                environmentalFactor,
                recommendation: this.getDiseaseRecommendation(topPrediction.disease, finalConfidence),
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            this.logger.error('Disease prediction failed:', error);
            throw error;
        }
    }

    async preprocessImageForCNN(imageData) {
        // Convert image data to tensor format expected by CNN
        // This is a simplified preprocessing - in production, you'd use more sophisticated methods
        const tensor = tf.tensor4d([imageData], [1, 224, 224, 3]);
        return tensor.div(255.0); // Normalize to [0, 1]
    }

    getDiseaseClasses() {
        return [
            'Healthy',
            'Powdery Mildew',
            'Downy Mildew',
            'Leaf Rust',
            'Bacterial Blight',
            'Viral Mosaic',
            'Anthracnose',
            'Black Spot',
            'Scab',
            'Root Rot'
        ];
    }

    getTopPrediction(probabilities, classes) {
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        return {
            disease: classes[maxIndex],
            confidence: probabilities[maxIndex]
        };
    }

    calculateEnvironmentalFactor(environmentalData) {
        // Adjust prediction confidence based on environmental conditions
        let factor = 1.0;
        
        if (environmentalData.humidity > 80) {
            factor *= 1.2; // Higher humidity increases fungal disease risk
        }
        
        if (environmentalData.temperature > 30) {
            factor *= 1.1; // High temperature can stress plants
        }
        
        if (environmentalData.leafWetness > 6) {
            factor *= 1.3; // Leaf wetness promotes disease
        }
        
        return Math.min(factor, 1.5); // Cap at 1.5x
    }

    calculateSeverity(confidence) {
        if (confidence > 0.8) return 'High';
        if (confidence > 0.6) return 'Medium';
        return 'Low';
    }

    getDiseaseRecommendation(disease, confidence) {
        const recommendations = {
            'Powdery Mildew': 'Apply fungicide spray and improve air circulation',
            'Downy Mildew': 'Reduce humidity and apply copper-based fungicide',
            'Leaf Rust': 'Apply systemic fungicide and remove affected leaves',
            'Bacterial Blight': 'Apply bactericide and avoid overhead watering',
            'Viral Mosaic': 'Remove infected plants and control insect vectors',
            'Anthracnose': 'Improve drainage and apply preventive fungicide',
            'Black Spot': 'Increase air circulation and apply fungicide',
            'Scab': 'Apply preventive fungicide during wet periods',
            'Root Rot': 'Improve drainage and reduce watering frequency'
        };
        
        const baseRecommendation = recommendations[disease] || 'Monitor closely and consult agricultural expert';
        
        if (confidence > 0.8) {
            return `URGENT: ${baseRecommendation}`;
        } else if (confidence > 0.6) {
            return `RECOMMENDED: ${baseRecommendation}`;
        } else {
            return `MONITOR: ${baseRecommendation}`;
        }
    }

    // Stress Prediction using LSTM
    async predictStress(timeSeriesData, environmentalData) {
        try {
            // Prepare time series data for LSTM
            const processedData = this.preprocessTimeSeriesData(timeSeriesData);
            
            // Make prediction
            const prediction = this.models.stressDetection.predict(processedData);
            const probabilities = await prediction.data();
            
            const stressLevels = ['Low', 'Medium', 'High'];
            const topPrediction = this.getTopPrediction(probabilities, stressLevels);
            
            // Analyze stress factors
            const stressFactors = this.analyzeStressFactors(timeSeriesData, environmentalData);
            
            return {
                stressLevel: topPrediction.disease, // Using same structure
                confidence: topPrediction.confidence,
                stressType: stressFactors.primaryStressor,
                factors: stressFactors.factors,
                timeToAction: this.calculateTimeToAction(topPrediction.confidence),
                recommendation: this.getStressRecommendation(stressFactors.primaryStressor),
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            this.logger.error('Stress prediction failed:', error);
            throw error;
        }
    }

    preprocessTimeSeriesData(timeSeriesData) {
        // Convert time series to tensor format for LSTM
        // Normalize and reshape data
        const normalizedData = timeSeriesData.map(sequence => 
            sequence.map(value => (value - 0.5) / 0.5) // Simple normalization
        );
        
        return tf.tensor3d([normalizedData], [1, normalizedData.length, normalizedData[0].length]);
    }

    analyzeStressFactors(timeSeriesData, environmentalData) {
        const factors = [];
        let primaryStressor = 'Unknown';
        
        // Analyze soil moisture trends
        const moistureTrend = this.analyzeTrend(timeSeriesData.map(d => d.soilMoisture || 0));
        if (moistureTrend.slope < -0.1) {
            factors.push('Declining soil moisture');
            primaryStressor = 'Water';
        }
        
        // Analyze temperature stress
        const avgTemp = this.calculateMean(timeSeriesData.map(d => d.temperature || 25));
        if (avgTemp > 35) {
            factors.push('High temperature stress');
            primaryStressor = 'Heat';
        }
        
        // Analyze NDVI trends
        const ndviTrend = this.analyzeTrend(timeSeriesData.map(d => d.ndvi || 0.7));
        if (ndviTrend.slope < -0.05) {
            factors.push('Declining vegetation health');
            if (primaryStressor === 'Unknown') primaryStressor = 'Nutrient';
        }
        
        return { factors, primaryStressor };
    }

    analyzeTrend(data) {
        // Simple linear regression to analyze trend
        const n = data.length;
        const x = Array.from({ length: n }, (_, i) => i);
        
        const sumX = x.reduce((sum, val) => sum + val, 0);
        const sumY = data.reduce((sum, val) => sum + val, 0);
        const sumXY = x.reduce((sum, val, i) => sum + val * data[i], 0);
        const sumXX = x.reduce((sum, val) => sum + val * val, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        return { slope, intercept };
    }

    calculateTimeToAction(confidence) {
        if (confidence > 0.8) return '24-48 hours';
        if (confidence > 0.6) return '2-3 days';
        return '1 week';
    }

    getStressRecommendation(stressType) {
        const recommendations = {
            'Water': 'Increase irrigation frequency and check soil moisture levels',
            'Heat': 'Provide shade cover and increase irrigation during hot periods',
            'Nutrient': 'Apply balanced fertilizer and test soil nutrient levels',
            'Light': 'Prune surrounding vegetation to improve light access',
            'Unknown': 'Monitor all environmental factors and consult expert'
        };
        
        return recommendations[stressType] || recommendations['Unknown'];
    }

    // Yield Prediction
    async predictYield(fieldData, historicalData) {
        try {
            // Prepare input features
            const features = this.prepareYieldFeatures(fieldData, historicalData);
            
            // Make prediction
            const prediction = this.models.yieldPrediction.predict(features);
            const yieldValue = await prediction.data();
            
            // Calculate yield increase/decrease percentage
            const historicalAverage = this.calculateHistoricalAverage(historicalData);
            const predictedYield = yieldValue[0];
            const yieldChange = ((predictedYield - historicalAverage) / historicalAverage) * 100;
            
            // Calculate confidence based on data quality
            const confidence = this.calculateYieldConfidence(fieldData, historicalData);
            
            return {
                predictedYield,
                yieldChange,
                yieldPercentage: yieldChange,
                confidence,
                factors: this.identifyYieldFactors(fieldData),
                recommendation: this.getYieldRecommendation(yieldChange),
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            this.logger.error('Yield prediction failed:', error);
            throw error;
        }
    }

    prepareYieldFeatures(fieldData, historicalData) {
        // Prepare 15 features for yield prediction model
        const features = [
            fieldData.avgNDVI || 0.7,
            fieldData.avgSoilMoisture || 70,
            fieldData.avgTemperature || 25,
            fieldData.totalRainfall || 500,
            fieldData.daysOfStress || 0,
            fieldData.diseaseIncidence || 0,
            fieldData.nutrientScore || 0.8,
            fieldData.plantDensity || 1.0,
            historicalData.avgYield || 100,
            historicalData.weatherVariability || 0.5,
            fieldData.soilQuality || 0.8,
            fieldData.irrigationFrequency || 3,
            fieldData.pestPressure || 0.2,
            fieldData.fieldAge || 5,
            fieldData.managementScore || 0.9
        ];
        
        return tf.tensor2d([features], [1, 15]);
    }

    calculateHistoricalAverage(historicalData) {
        if (!historicalData.yields || historicalData.yields.length === 0) {
            return 100; // Default yield value
        }
        
        return this.calculateMean(historicalData.yields);
    }

    calculateYieldConfidence(fieldData, historicalData) {
        let confidence = 0.5; // Base confidence
        
        // Increase confidence based on data completeness
        if (fieldData.avgNDVI) confidence += 0.1;
        if (fieldData.avgSoilMoisture) confidence += 0.1;
        if (fieldData.totalRainfall) confidence += 0.1;
        if (historicalData.yields && historicalData.yields.length > 3) confidence += 0.2;
        
        return Math.min(confidence, 0.95);
    }

    identifyYieldFactors(fieldData) {
        const factors = [];
        
        if (fieldData.avgNDVI > 0.8) {
            factors.push('Excellent vegetation health');
        } else if (fieldData.avgNDVI < 0.6) {
            factors.push('Vegetation stress detected');
        }
        
        if (fieldData.avgSoilMoisture > 80) {
            factors.push('Optimal soil moisture');
        } else if (fieldData.avgSoilMoisture < 50) {
            factors.push('Water stress risk');
        }
        
        if (fieldData.diseaseIncidence > 0.3) {
            factors.push('Disease pressure affecting yield');
        }
        
        if (fieldData.nutrientScore > 0.8) {
            factors.push('Good nutrient management');
        }
        
        return factors;
    }

    getYieldRecommendation(yieldChange) {
        if (yieldChange > 5) {
            return 'Continue current management practices - excellent performance expected';
        } else if (yieldChange > 0) {
            return 'Minor improvements in irrigation or nutrition could boost yield further';
        } else if (yieldChange > -5) {
            return 'Address stress factors and improve crop management to maintain yield';
        } else {
            return 'Urgent intervention needed - significant yield loss predicted';
        }
    }

    // Health Map Generation
    async generateHealthMap(indices, anomalies) {
        try {
            const healthMap = {
                overallScore: this.calculateOverallHealthScore(indices),
                zoneHealth: this.calculateZoneHealth(indices, anomalies),
                riskAreas: this.identifyRiskAreas(anomalies),
                recommendations: this.generateHealthRecommendations(indices, anomalies),
                timestamp: new Date().toISOString()
            };
            
            return healthMap;
        } catch (error) {
            this.logger.error('Health map generation failed:', error);
            throw error;
        }
    }

    calculateOverallHealthScore(indices) {
        // Combine multiple vegetation indices for overall score
        const weights = {
            ndvi: 0.4,
            evi: 0.3,
            savi: 0.2,
            ndre: 0.1
        };
        
        let weightedSum = 0;
        let totalWeight = 0;
        
        Object.keys(weights).forEach(index => {
            if (indices[index] !== null && indices[index] !== undefined) {
                weightedSum += indices[index] * weights[index];
                totalWeight += weights[index];
            }
        });
        
        return totalWeight > 0 ? (weightedSum / totalWeight) * 100 : 0;
    }

    calculateZoneHealth(indices, anomalies) {
        // Simulate zone-based health calculation
        const zones = [];
        
        for (let i = 1; i <= 5; i++) {
            const zoneAnomalies = anomalies.anomalies?.filter(a => 
                Math.floor(a.pixelIndex / 1000) === i - 1
            ) || [];
            
            let healthScore = this.calculateOverallHealthScore(indices);
            
            // Reduce health score based on anomalies in zone
            const anomalyPenalty = zoneAnomalies.length * 5;
            healthScore = Math.max(0, healthScore - anomalyPenalty);
            
            zones.push({
                zone: i,
                health: healthScore / 100,
                status: this.getHealthStatus(healthScore),
                anomalyCount: zoneAnomalies.length
            });
        }
        
        return zones;
    }

    getHealthStatus(score) {
        if (score >= 85) return 'excellent';
        if (score >= 70) return 'good';
        if (score >= 55) return 'fair';
        return 'poor';
    }

    identifyRiskAreas(anomalies) {
        if (!anomalies.anomalies) return [];
        
        const riskAreas = [];
        const highRiskAnomalies = anomalies.anomalies.filter(a => a.severity === 'high');
        
        // Group anomalies by location/area
        const areaGroups = this.groupAnomaliesByArea(highRiskAnomalies);
        
        areaGroups.forEach((group, index) => {
            riskAreas.push({
                areaId: index + 1,
                riskLevel: 'high',
                anomalyCount: group.length,
                primaryRisk: this.identifyPrimaryRisk(group),
                coordinates: this.calculateAreaCoordinates(group)
            });
        });
        
        return riskAreas;
    }

    groupAnomaliesByArea(anomalies) {
        // Simple spatial grouping based on pixel proximity
        const groups = [];
        const processed = new Set();
        
        anomalies.forEach(anomaly => {
            if (processed.has(anomaly.pixelIndex)) return;
            
            const group = [anomaly];
            processed.add(anomaly.pixelIndex);
            
            // Find nearby anomalies (simplified)
            anomalies.forEach(other => {
                if (!processed.has(other.pixelIndex) && 
                    Math.abs(other.pixelIndex - anomaly.pixelIndex) < 100) {
                    group.push(other);
                    processed.add(other.pixelIndex);
                }
            });
            
            if (group.length >= 3) { // Minimum group size
                groups.push(group);
            }
        });
        
        return groups;
    }

    identifyPrimaryRisk(anomalyGroup) {
        const riskCounts = {};
        
        anomalyGroup.forEach(anomaly => {
            const riskType = this.categorizeAnomalyRisk(anomaly);
            riskCounts[riskType] = (riskCounts[riskType] || 0) + 1;
        });
        
        return Object.keys(riskCounts).reduce((a, b) => 
            riskCounts[a] > riskCounts[b] ? a : b
        );
    }

    categorizeAnomalyRisk(anomaly) {
        if (anomaly.description.includes('disease')) return 'disease';
        if (anomaly.description.includes('stress')) return 'stress';
        if (anomaly.description.includes('nutrient')) return 'nutrient';
        return 'unknown';
    }

    calculateAreaCoordinates(anomalyGroup) {
        // Convert pixel indices to approximate coordinates
        // This is simplified - in production, you'd have proper geospatial calculations
        const avgPixelIndex = this.calculateMean(anomalyGroup.map(a => a.pixelIndex));
        
        // Simulate coordinate conversion
        const lat = 40.7128 + (avgPixelIndex % 1000) * 0.0001;
        const lng = -74.0060 + Math.floor(avgPixelIndex / 1000) * 0.0001;
        
        return { lat, lng };
    }

    generateHealthRecommendations(indices, anomalies) {
        const recommendations = [];
        
        // NDVI-based recommendations
        if (indices.ndvi < 0.6) {
            recommendations.push({
                priority: 'high',
                category: 'vegetation_health',
                action: 'Investigate low NDVI areas for stress factors',
                details: 'NDVI below 0.6 indicates vegetation stress'
            });
        }
        
        // Anomaly-based recommendations
        if (anomalies.anomalies && anomalies.anomalies.length > 10) {
            recommendations.push({
                priority: 'medium',
                category: 'anomaly_management',
                action: 'Increase monitoring frequency in anomaly areas',
                details: `${anomalies.anomalies.length} anomalies detected`
            });
        }
        
        // EVI-based recommendations
        if (indices.evi && indices.evi < 0.5) {
            recommendations.push({
                priority: 'medium',
                category: 'canopy_management',
                action: 'Assess canopy density and chlorophyll content',
                details: 'Low EVI suggests reduced photosynthetic activity'
            });
        }
        
        return recommendations;
    }

    // Scheduled Analysis
    async runScheduledAnalysis() {
        try {
            this.logger.info('Running scheduled AI analysis...');
            
            // Get all active fields
            const fields = await this.getActiveFields();
            
            for (const field of fields) {
                await this.processFieldAnalysis(field);
            }
            
            this.logger.info('Scheduled analysis completed');
        } catch (error) {
            this.logger.error('Scheduled analysis failed:', error);
            throw error;
        }
    }

    async processFieldAnalysis(field) {
        try {
            // Get latest sensor data
            const sensorData = await this.getSensorData(field._id);
            
            // Calculate field health
            const healthData = await this.calculateFieldHealth(field._id);
            
            // Check for stress conditions
            if (healthData.overallScore < 70) {
                await this.triggerStressAnalysis(field._id, healthData);
            }
            
            // Update field status
            await this.updateFieldStatus(field._id, healthData);
            
        } catch (error) {
            this.logger.error(`Field analysis failed for ${field._id}:`, error);
        }
    }

    async calculateFieldHealth(fieldId) {
        // Simulate field health calculation
        const baseHealth = 75 + Math.random() * 20;
        
        return {
            fieldId,
            overallScore: baseHealth,
            category: this.getHealthStatus(baseHealth),
            lastUpdated: new Date().toISOString()
        };
    }

    async getSensorData(fieldId) {
        // Simulate sensor data retrieval
        return {
            fieldId,
            soilMoisture: 60 + Math.random() * 30,
            temperature: 20 + Math.random() * 15,
            humidity: 50 + Math.random() * 40,
            timestamp: new Date().toISOString()
        };
    }

    async getActiveFields() {
        // Simulate active fields retrieval
        return [
            { _id: 'field1', name: 'North Field' },
            { _id: 'field2', name: 'South Field' },
            { _id: 'field3', name: 'East Field' }
        ];
    }

    async triggerStressAnalysis(fieldId, healthData) {
        this.logger.info(`Triggering stress analysis for field ${fieldId}`);
        // Implement stress analysis logic
    }

    async updateFieldStatus(fieldId, healthData) {
        this.logger.info(`Updating status for field ${fieldId}: ${healthData.category}`);
        // Implement field status update logic
    }

    async storePrediction(type, prediction, fieldId) {
        // Store prediction in database
        this.logger.info(`Storing ${type} prediction for field ${fieldId}`);
        // Implement database storage logic
    }

    async generateDailyReport(fieldId) {
        // Generate comprehensive daily report
        this.logger.info(`Generating daily report for field ${fieldId}`);
        
        return {
            fieldId,
            date: new Date().toISOString().split('T')[0],
            summary: {
                overallHealth: 85,
                alerts: 2,
                predictions: 3
            },
            generatedAt: new Date().toISOString()
        };
    }
}

module.exports = AIService;
