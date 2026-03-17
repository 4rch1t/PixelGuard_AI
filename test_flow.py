import requests

# Test 1: Upload image
test_image = r'test\REAL\0291.jpg'
with open(test_image, 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/predict', files=files)

if response.status_code == 200:
    data = response.json()
    print('✓ Image analysis working')
    print('  - Prediction:', data['prediction'])
    print('  - Confidence: {:.1f}%'.format(data['confidence']*100))
    print('  - Analysis ID:', data['analysis_id'][:8] + '...')
    
    # Test 2: Generate PDF
    pdf_response = requests.get('http://localhost:5000/generate-report/{}'.format(data['analysis_id']))
    if pdf_response.status_code == 200:
        print('✓ PDF generation working')
        print('  - PDF size: {} bytes'.format(len(pdf_response.content)))
    else:
        print('✗ PDF generation failed')
else:
    print('✗ Image analysis failed:', response.status_code)
