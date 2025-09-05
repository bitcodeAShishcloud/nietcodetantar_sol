const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const dotenv = require('dotenv');
const mongoose = require('mongoose');
const http = require('http');
const socketIo = require('socket.io');
const cron = require('node-cron');
const winston = require('winston');

// Load environment variables
dotenv.config();

// Import routes
const authRoutes = require('./routes/auth');
const fieldRoutes = require('./routes/fields');
const sensorRoutes = require('./routes/sensors');
const imageRoutes = require('./routes/images');
const aiRoutes = require('./routes/ai');
const alertRoutes = require('./routes/alerts');

// Import services
const AIService = require('./services/aiService');
const SensorService = require('./services/sensorService');
const ImageService = require('./services/imageService');
const AlertService = require('./services/alertService');

// Configure logger
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
        new winston.transports.File({ filename: 'logs/combined.log' }),
        new winston.transports.Console({
            format: winston.format.simple()
        })
    ]
});

class AgricultureServer {
    constructor() {
        this.app = express();
        this.server = http.createServer(this.app);
        this.io = socketIo(this.server, {
            cors: {
                origin: "*",
                methods: ["GET", "POST"]
            }
        });
        
        this.port = process.env.PORT || 3000;
        this.apiKey = process.env.GOOGLE_API_KEY || 'AIzaSyApIAX0hMK5LKxEu5-pR590MaTw1YJ5Nsk';
        
        this.initializeServices();
        this.setupMiddleware();
        this.setupRoutes();
        this.setupSocketHandlers();
        this.setupCronJobs();
        this.connectDatabase();
    }

    initializeServices() {
        this.aiService = new AIService(this.apiKey);
        this.sensorService = new SensorService();
        this.imageService = new ImageService();
        this.alertService = new AlertService(this.io);
    }

    setupMiddleware() {
        // Security middleware
        this.app.use(helmet());
        
        // CORS configuration
        this.app.use(cors({
            origin: process.env.FRONTEND_URL || '*',
            credentials: true
        }));

        // Body parsing middleware
        this.app.use(express.json({ limit: '50mb' }));
        this.app.use(express.urlencoded({ extended: true, limit: '50mb' }));

        // Logging middleware
        this.app.use((req, res, next) => {
            logger.info(`${req.method} ${req.path}`, { 
                ip: req.ip, 
                userAgent: req.get('User-Agent') 
            });
            next();
        });

        // Static files
        this.app.use('/uploads', express.static('uploads'));
    }

    setupRoutes() {
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({ 
                status: 'healthy', 
                timestamp: new Date().toISOString(),
                version: process.env.npm_package_version || '1.0.0'
            });
        });

        // API routes
        this.app.use('/api/auth', authRoutes);
        this.app.use('/api/fields', fieldRoutes);
        this.app.use('/api/sensors', sensorRoutes);
        this.app.use('/api/images', imageRoutes);
        this.app.use('/api/ai', aiRoutes);
        this.app.use('/api/alerts', alertRoutes);

        // Real-time data endpoints
        this.app.get('/api/realtime/sensors', this.handleRealtimeSensors.bind(this));
        this.app.get('/api/realtime/health', this.handleRealtimeHealth.bind(this));
        
        // Hyperspectral analysis endpoint
        this.app.post('/api/analysis/hyperspectral', this.handleHyperspectralAnalysis.bind(this));
        
        // AI prediction endpoints
        this.app.post('/api/predictions/disease', this.handleDiseaseprediction.bind(this));
        this.app.post('/api/predictions/yield', this.handleYieldPrediction.bind(this));
        this.app.post('/api/predictions/stress', this.handleStressPrediction.bind(this));

        // Error handling middleware
        this.app.use(this.errorHandler);

        // 404 handler
        this.app.use('*', (req, res) => {
            res.status(404).json({ 
                error: 'Endpoint not found',
                path: req.originalUrl 
            });
        });
    }

    setupSocketHandlers() {
        this.io.on('connection', (socket) => {
            logger.info(`Client connected: ${socket.id}`);

            socket.on('subscribe_field', (fieldId) => {
                socket.join(`field_${fieldId}`);
                logger.info(`Client ${socket.id} subscribed to field ${fieldId}`);
            });

            socket.on('unsubscribe_field', (fieldId) => {
                socket.leave(`field_${fieldId}`);
                logger.info(`Client ${socket.id} unsubscribed from field ${fieldId}`);
            });

            socket.on('disconnect', () => {
                logger.info(`Client disconnected: ${socket.id}`);
            });
        });
    }

    setupCronJobs() {
        // Collect sensor data every 5 minutes
        cron.schedule('*/5 * * * *', async () => {
            try {
                await this.sensorService.collectAllSensorData();
                logger.info('Sensor data collection completed');
            } catch (error) {
                logger.error('Sensor data collection failed:', error);
            }
        });

        // Run AI analysis every 15 minutes
        cron.schedule('*/15 * * * *', async () => {
            try {
                await this.aiService.runScheduledAnalysis();
                logger.info('Scheduled AI analysis completed');
            } catch (error) {
                logger.error('Scheduled AI analysis failed:', error);
            }
        });

        // Generate daily reports at 6 AM
        cron.schedule('0 6 * * *', async () => {
            try {
                await this.generateDailyReports();
                logger.info('Daily reports generated');
            } catch (error) {
                logger.error('Daily report generation failed:', error);
            }
        });
    }

    async connectDatabase() {
        try {
            const mongoUrl = process.env.MONGODB_URL || 'mongodb://localhost:27017/agricultural-ai';
            await mongoose.connect(mongoUrl, {
                useNewUrlParser: true,
                useUnifiedTopology: true
            });
            logger.info('Connected to MongoDB');
        } catch (error) {
            logger.error('MongoDB connection failed:', error);
            process.exit(1);
        }
    }

    // Real-time endpoint handlers
    async handleRealtimeSensors(req, res) {
        try {
            const { fieldId } = req.query;
            const sensorData = await this.sensorService.getLatestSensorData(fieldId);
            
            // Emit to connected clients
            this.io.to(`field_${fieldId}`).emit('sensor_update', sensorData);
            
            res.json(sensorData);
        } catch (error) {
            logger.error('Realtime sensors error:', error);
            res.status(500).json({ error: 'Failed to fetch sensor data' });
        }
    }

    async handleRealtimeHealth(req, res) {
        try {
            const { fieldId } = req.query;
            const healthData = await this.aiService.calculateFieldHealth(fieldId);
            
            // Emit to connected clients
            this.io.to(`field_${fieldId}`).emit('health_update', healthData);
            
            res.json(healthData);
        } catch (error) {
            logger.error('Realtime health error:', error);
            res.status(500).json({ error: 'Failed to fetch health data' });
        }
    }

    // Hyperspectral analysis handler
    async handleHyperspectralAnalysis(req, res) {
        try {
            const { imageData, metadata } = req.body;
            
            logger.info('Starting hyperspectral analysis');
            
            // Process hyperspectral image
            const processedImage = await this.imageService.processHyperspectralImage(imageData, metadata);
            
            // Extract vegetation indices
            const indices = await this.aiService.calculateVegetationIndices(processedImage);
            
            // Detect anomalies
            const anomalies = await this.aiService.detectSpectralAnomalies(processedImage);
            
            // Generate health map
            const healthMap = await this.aiService.generateHealthMap(indices, anomalies);
            
            // Create alert if necessary
            if (anomalies.length > 0) {
                await this.alertService.checkAndCreateAlerts(anomalies);
            }

            const result = {
                indices,
                anomalies,
                healthMap,
                timestamp: new Date().toISOString(),
                processingTime: Date.now() - req.startTime
            };

            res.json(result);
            
            logger.info('Hyperspectral analysis completed');
        } catch (error) {
            logger.error('Hyperspectral analysis error:', error);
            res.status(500).json({ error: 'Analysis failed' });
        }
    }

    // AI prediction handlers
    async handleDiseasePredicti on(req, res) {
        try {
            const { imageData, environmentalData, fieldId } = req.body;
            
            const prediction = await this.aiService.predictDisease(imageData, environmentalData);
            
            // Store prediction in database
            await this.aiService.storePrediction('disease', prediction, fieldId);
            
            // Create alert if high risk
            if (prediction.confidence > 0.7) {
                await this.alertService.createDiseaseAlert(prediction, fieldId);
            }

            res.json(prediction);
        } catch (error) {
            logger.error('Disease prediction error:', error);
            res.status(500).json({ error: 'Disease prediction failed' });
        }
    }

    async handleYieldPrediction(req, res) {
        try {
            const { fieldData, historicalData, fieldId } = req.body;
            
            const prediction = await this.aiService.predictYield(fieldData, historicalData);
            
            await this.aiService.storePrediction('yield', prediction, fieldId);

            res.json(prediction);
        } catch (error) {
            logger.error('Yield prediction error:', error);
            res.status(500).json({ error: 'Yield prediction failed' });
        }
    }

    async handleStressPrediction(req, res) {
        try {
            const { timeSeriesData, environmentalData, fieldId } = req.body;
            
            const prediction = await this.aiService.predictStress(timeSeriesData, environmentalData);
            
            await this.aiService.storePrediction('stress', prediction, fieldId);
            
            // Create alert if stress detected
            if (prediction.stressLevel === 'High') {
                await this.alertService.createStressAlert(prediction, fieldId);
            }

            res.json(prediction);
        } catch (error) {
            logger.error('Stress prediction error:', error);
            res.status(500).json({ error: 'Stress prediction failed' });
        }
    }

    async generateDailyReports() {
        // Generate comprehensive daily reports for all fields
        const fields = await this.getActiveFields();
        
        for (const field of fields) {
            const report = await this.aiService.generateDailyReport(field._id);
            
            // Send report notifications
            this.io.to(`field_${field._id}`).emit('daily_report', report);
            
            logger.info(`Daily report generated for field ${field._id}`);
        }
    }

    async getActiveFields() {
        const Field = require('./models/Field');
        return await Field.find({ status: 'active' });
    }

    // Error handling middleware
    errorHandler(err, req, res, next) {
        logger.error('Unhandled error:', err);

        if (err.name === 'ValidationError') {
            return res.status(400).json({ 
                error: 'Validation error', 
                details: err.message 
            });
        }

        if (err.name === 'CastError') {
            return res.status(400).json({ 
                error: 'Invalid ID format' 
            });
        }

        res.status(500).json({ 
            error: 'Internal server error',
            message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
        });
    }

    start() {
        this.server.listen(this.port, () => {
            logger.info(`Agricultural AI Server running on port ${this.port}`);
            logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
        });

        // Graceful shutdown
        process.on('SIGTERM', () => {
            logger.info('SIGTERM received, shutting down gracefully');
            this.server.close(() => {
                mongoose.connection.close();
                process.exit(0);
            });
        });
    }
}

// Start the server
if (require.main === module) {
    const server = new AgricultureServer();
    server.start();
}

module.exports = AgricultureServer;
