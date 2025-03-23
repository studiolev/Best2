# LEVisions - Furniture Detection App

A Streamlit web application for detecting furniture in interior images.

## Features

- Upload interior images
- Detect furniture items
- Display detection results with confidence scores
- Find similar furniture recommendations

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/furniture-detector.git
cd furniture-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app/app.py
```

The app will be available at http://localhost:8501

## Requirements

- Python 3.10+
- Streamlit
- Pillow
- Requests

## Project Structure

```
furniture-detector/
├── streamlit_app/
│   ├── app.py
│   └── streamlit_UI_code.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 