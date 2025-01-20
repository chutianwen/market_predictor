import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path
import markdown
import numpy as np
import warnings
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

class MarketPredictionReport:
    def __init__(self):
        self.MODEL_DIR = 'market_models'
        self.DATA_DIR = 'market_data'
        self.REPORT_DIR = 'market_reports'
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        for dir_path in [self.MODEL_DIR, self.DATA_DIR, self.REPORT_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Creating directory: {dir_path}")
                
    def get_latest_files(self):
        """Get the latest data and prediction files"""
        # Get latest prediction results
        prediction_files = [f for f in os.listdir(self.MODEL_DIR) 
                          if f.startswith('prediction_results_') and f.endswith('.csv')]
        if not prediction_files:
            raise FileNotFoundError("No prediction files found in market_models directory")
        latest_prediction = sorted(prediction_files)[-1]
        
        # Get latest data file
        data_files = [f for f in os.listdir(self.DATA_DIR) 
                     if f.startswith('market_data_') and f.endswith('.csv')]
        if not data_files:
            raise FileNotFoundError("No data files found in market_data directory")
        latest_data = sorted(data_files)[-1]
        
        return latest_prediction, latest_data
    
    def create_performance_plot(self, feature_importance_spy, feature_importance_qqq):
        """Create feature importance plots"""
        plt.figure(figsize=(12, 6))
        
        # Set font for plots
        plt.rcParams['font.family'] = 'Times New Roman'
        
        # SPY feature importance
        plt.subplot(1, 2, 1)
        sns.barplot(x='importance', y='feature', data=feature_importance_spy.head(5))
        plt.title('SPY - Top 5 Important Features')
        
        # QQQ feature importance
        plt.subplot(1, 2, 2)
        sns.barplot(x='importance', y='feature', data=feature_importance_qqq.head(5))
        plt.title('QQQ - Top 5 Important Features')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.REPORT_DIR, 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_pdf(self, markdown_content, pdf_path, plot_path):
        """Generate PDF using ReportLab"""
        try:
            # Create PDF document
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            styles = getSampleStyleSheet()
            
            # Create custom styles
            styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30
            ))
            styles.add(ParagraphStyle(
                name='CustomHeading1',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=16
            ))
            styles.add(ParagraphStyle(
                name='CustomHeading2',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12
            ))
            
            # Create story (content)
            story = []
            
            # Process markdown content
            lines = markdown_content.split('\n')
            table_data = []
            in_table = False
            
            for line in lines:
                if line.strip() == '':
                    if in_table and table_data:
                        # Create and style the table
                        t = Table(table_data)
                        t.setStyle(TableStyle([
                            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),  # Header font
                            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),      # Body font
                            ('FONTSIZE', (0,0), (-1,-1), 10),
                            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                            ('ALIGN', (0,0), (-1,0), 'CENTER'),
                            ('BOTTOMPADDING', (0,0), (-1,0), 12),
                            ('TOPPADDING', (0,0), (-1,0), 12),
                            ('BOTTOMPADDING', (0,1), (-1,-1), 8),
                            ('TOPPADDING', (0,1), (-1,-1), 8),
                        ]))
                        story.append(t)
                        story.append(Spacer(1, 12))
                        table_data = []
                        in_table = False
                    else:
                        story.append(Spacer(1, 12))
                elif line.startswith('# '):
                    story.append(Paragraph(line[2:], styles['CustomTitle']))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], styles['CustomHeading1']))
                elif line.startswith('### '):
                    story.append(Paragraph(line[4:], styles['CustomHeading2']))
                elif line.startswith('- '):
                    story.append(Paragraph(f"• {line[2:]}", styles['Normal']))
                elif line.startswith('!['):
                    img = Image(plot_path, width=6*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                elif line.startswith('|'):
                    # Process table rows
                    cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    if cells:
                        if not in_table:
                            in_table = True
                        if '---' not in line:  # Skip separator line
                            table_data.append(cells)
                else:
                    if in_table and table_data:
                        # Create and style the table
                        t = Table(table_data)
                        t.setStyle(TableStyle([
                            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),  # Header font
                            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),      # Body font
                            ('FONTSIZE', (0,0), (-1,-1), 10),
                            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                            ('ALIGN', (0,0), (-1,0), 'CENTER'),
                            ('BOTTOMPADDING', (0,0), (-1,0), 12),
                            ('TOPPADDING', (0,0), (-1,0), 12),
                            ('BOTTOMPADDING', (0,1), (-1,-1), 8),
                            ('TOPPADDING', (0,1), (-1,-1), 8),
                        ]))
                        story.append(t)
                        story.append(Spacer(1, 12))
                        table_data = []
                        in_table = False
                    story.append(Paragraph(line, styles['Normal']))
            
            # Handle any remaining table
            if in_table and table_data:
                t = Table(table_data)
                t.setStyle(TableStyle([
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),  # Header font
                    ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),      # Body font
                    ('FONTSIZE', (0,0), (-1,-1), 10),
                    ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                    ('ALIGN', (0,0), (-1,0), 'CENTER'),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('TOPPADDING', (0,0), (-1,0), 12),
                    ('BOTTOMPADDING', (0,1), (-1,-1), 8),
                    ('TOPPADDING', (0,1), (-1,-1), 8),
                ]))
                story.append(t)
                story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            return True
        except Exception as e:
            warnings.warn(f"Could not generate PDF: {str(e)}")
            return False
    
    def generate_report(self):
        """Generate prediction report"""
        try:
            latest_prediction, latest_data = self.get_latest_files()
            
            # Read prediction results
            pred_df = pd.read_csv(os.path.join(self.MODEL_DIR, latest_prediction))
            data_df = pd.read_csv(os.path.join(self.DATA_DIR, latest_data))
            
            # Extract feature importance data
            feature_importance_spy = pd.DataFrame({
                'feature': [pred_df[f'SPY_Top{i+1}_Feature'].iloc[0] for i in range(5)],
                'importance': [pred_df[f'SPY_Top{i+1}_Importance'].iloc[0] for i in range(5)]
            })
            
            feature_importance_qqq = pd.DataFrame({
                'feature': [pred_df[f'QQQ_Top{i+1}_Feature'].iloc[0] for i in range(5)],
                'importance': [pred_df[f'QQQ_Top{i+1}_Importance'].iloc[0] for i in range(5)]
            })
            
            # Generate feature importance plot
            plot_path = self.create_performance_plot(feature_importance_spy, feature_importance_qqq)
            
            # Generate report content
            report_date = datetime.now().strftime("%Y-%m")
            target_month = pred_df['Target_Month'].iloc[0]
            
            markdown_content = f"""
# Market Prediction Monthly Report - {report_date}

## Prediction Overview

This report presents machine learning model predictions for SPY and QQQ returns for the next month ({target_month}).

### Prediction Results

- **SPY Expected Return**: {pred_df['SPY_Predicted_Return'].iloc[0]:.2f}%
  - Confidence Interval: {pred_df['SPY_Confidence_Lower'].iloc[0]:.2f}% to {pred_df['SPY_Confidence_Upper'].iloc[0]:.2f}%
  
- **QQQ Expected Return**: {pred_df['QQQ_Predicted_Return'].iloc[0]:.2f}%
  - Confidence Interval: {pred_df['QQQ_Confidence_Lower'].iloc[0]:.2f}% to {pred_df['QQQ_Confidence_Upper'].iloc[0]:.2f}%

## Prediction Methodology

We use a Random Forest Regression model with the following advantages:
1. Captures non-linear market relationships
2. Handles feature interactions automatically
3. Provides reliable uncertainty estimates
4. Reduces overfitting through ensemble learning

### Model Performance Metrics

- SPY Model R² Score: {pred_df['SPY_R2_Score'].iloc[0]:.4f}
- QQQ Model R² Score: {pred_df['QQQ_R2_Score'].iloc[0]:.4f}

## Key Influencing Factors

![Feature Importance Analysis](feature_importance.png)

### SPY Key Factors
{feature_importance_spy.to_markdown()}

### QQQ Key Factors
{feature_importance_qqq.to_markdown()}

## Market Environment Analysis

Based on latest economic indicators:

- GDP: {data_df['GDP'].iloc[-1]:.2f}
- CPI: {data_df['CPI'].iloc[-1]:.2f}
- Unemployment Rate: {data_df['Unemployment'].iloc[-1]:.2f}%
- PMI: {data_df['PMI'].iloc[-1]:.2f}

## Risk Disclaimer

1. Predictions are based on historical data and current market conditions
2. Confidence intervals represent the range of uncertainty
3. Use these predictions as one of many inputs for investment decisions

## Methodology Details

### Data Sources
- Economic Indicators: FRED (Federal Reserve Economic Data)
- Market Data: Yahoo Finance

### Feature Engineering
- Year-over-Year and Month-over-Month changes in economic indicators
- Seasonal factors (monthly and quarterly features)
- Historical market performance data

### Model Validation
- Cross-validation for performance evaluation
- Ensemble methods for prediction intervals
- Regular model updates to adapt to market changes

---
*This report is algorithmically generated for reference only. Investment involves risks.*
"""
            
            # Save as markdown file
            report_path = os.path.join(self.REPORT_DIR, f'market_prediction_report_{report_date}.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"\nReports generated:")
            print(f"1. Markdown version: {report_path}")
            
            # Generate PDF
            pdf_path = os.path.join(self.REPORT_DIR, f'market_prediction_report_{report_date}.pdf')
            if self.generate_pdf(markdown_content, pdf_path, plot_path):
                print(f"2. PDF version: {pdf_path}")
            else:
                print("\nNote: PDF version could not be generated.")
            
            return report_path
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            raise

if __name__ == "__main__":
    report_generator = MarketPredictionReport()
    report_generator.generate_report() 