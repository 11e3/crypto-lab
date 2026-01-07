"""Create Jupyter notebooks with proper JSON format using nbformat."""

import nbformat as nbf


def create_backtesting_notebook():
    """Create 01-Backtesting-Case-Study.ipynb"""
    nb = nbf.v4.new_notebook()

    # Cell 1: Title
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "# 백테스팅 사례 연구: 변동성 돌파 전략\n\n"
            "이 노트북은 crypto-quant-system을 사용하여 변동성 돌파(Volatility Breakout) 전략을 백테스팅하는 과정을 보여줍니다."
        )
    )

    # Cell 2: Environment Setup
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. 환경 설정 및 필수 라이브러리\n\n"
            "필요한 라이브러리를 import하고 설정을 초기화합니다."
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "import sys\nfrom pathlib import Path\n\n"
            "# Add project root to path\nproject_root = Path.cwd()\nif str(project_root) not in sys.path:\n    sys.path.insert(0, str(project_root))\n\n"
            "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom datetime import datetime, timedelta\nimport warnings\nwarnings.filterwarnings('ignore')\n\n"
            "# Import from crypto-quant-system\nfrom src.backtester.engine import BacktestEngine\nfrom src.strategies.volatility_breakout import VanillaVBO\nfrom src.data.collector_factory import CollectorFactory\nfrom src.config.loader import load_config\n\n"
            "# Set up visualization\nplt.style.use('seaborn-v0_8-darkgrid')\nsns.set_palette('husl')\n\nprint('✓ 모든 라이브러리가 성공적으로 로드되었습니다.')"
        )
    )

    # Cell 3: Data Collection
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. 데이터 수집 및 준비\n\nUpbit 거래소에서 BTC/KRW 데이터를 수집합니다."
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Load configuration\nconfig = load_config()\n\n"
            "# Collect data\ncollector = CollectorFactory.create('upbit', 'BTC/KRW')\ndf = collector.collect(days=365)\n\n"
            "# Display basic information\nprint(f'Data Shape: {df.shape}')\nprint(f'Date Range: {df.index[0]} to {df.index[-1]}')\nprint(f'\\nFirst few rows:')\nprint(df.head())\nprint(f'\\nData Info:')\nprint(df.info())"
        )
    )

    # Cell 4: Strategy Configuration
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. 전략 설정 및 백테스팅 실행\n\n변동성 돌파 전략을 설정하고 백테스팅을 실행합니다."
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Create strategy\nstrategy = VanillaVBO(\n    name='Volatility Breakout Strategy',\n    lookback_period=20,\n    atr_period=10,\n    volatility_threshold=1.0\n)\n\n# Initialize backtester\nbacktester = BacktestEngine(\n    initial_balance=10_000_000,  # 1000만 원\n    trading_fees=0.0005,  # 0.05%\n    slippage=0.001  # 0.1%\n)\n\n# Run backtest\nresults = backtester.run(df, strategy)\n\nprint('✓ 백테스팅이 완료되었습니다.')\nprint(f'\\n성과 요약:')\nprint(f'- 총 거래 수: {results[\"total_trades\"]}')\nprint(f'- 승리 거래: {results[\"winning_trades\"]}')\nprint(f'- 최종 수익: {results[\"total_return\"]:.2%}')\nprint(f'- 샤프 비율: {results[\"sharpe_ratio\"]:.2f}')"
        )
    )

    # Cell 5: Performance Metrics
    nb.cells.append(
        nbf.v4.new_markdown_cell("## 4. 성과 지표 분석\n\n전략의 성과를 다양한 지표로 분석합니다.")
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "import pandas as pd\n\n"
            "# Create metrics summary\nmetrics = {\n    '총 수익률': f\"{results['total_return']:.2%}\",\n    '연율화 수익률 (CAGR)': f\"{results['cagr']:.2%}\",\n    '샤프 비율': f\"{results['sharpe_ratio']:.2f}\",\n    '소르티노 비율': f\"{results['sortino_ratio']:.2f}\",\n    '칼마 비율': f\"{results['calmar_ratio']:.2f}\",\n    '최대 낙폭': f\"{results['max_drawdown']:.2%}\",\n    '승률': f\"{results['win_rate']:.2%}\",\n    '평균 거래 수익': f\"{results['avg_trade_return']:.2%}\"\n}\n\nmetrics_df = pd.DataFrame(metrics.items(), columns=['지표', '값'])\nprint(metrics_df.to_string(index=False))"
        )
    )

    # Cell 6: Equity Curve
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. 자산 곡선 시각화\n\n시간에 따른 포트폴리오 가치의 변화를 시각화합니다."
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(14, 6))\n\n"
            "# Plot equity curve\nax.plot(results['dates'], results['equity'], linewidth=2, label='포트폴리오 가치')\nax.fill_between(results['dates'], results['equity'], alpha=0.3)\n\n"
            "ax.set_xlabel('날짜')\nax.set_ylabel('포트폴리오 가치 (원)')\nax.set_title('자산 곡선: 변동성 돌파 전략')\nax.legend()\nax.grid(True, alpha=0.3)\nplt.xticks(rotation=45)\nplt.tight_layout()\nplt.show()\n\nprint('✓ 자산 곡선 표시 완료')"
        )
    )

    # Cell 7: Monthly Returns Heatmap
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 6. 월별 수익률 히트맵\n\n월별/연도별 수익률을 히트맵으로 시각화합니다."
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Create monthly returns heatmap\nmonthly_returns = results['monthly_returns']\n\nfig, ax = plt.subplots(figsize=(12, 6))\nsns.heatmap(monthly_returns, annot=True, fmt='.1%', cmap='RdYlGn', center=0, \n            cbar_kws={'label': '수익률'}, ax=ax)\n\nax.set_title('월별 수익률 히트맵')\nax.set_xlabel('월')\nax.set_ylabel('연도')\nplt.tight_layout()\nplt.show()\n\nprint('✓ 월별 수익률 히트맵 표시 완료')"
        )
    )

    # Cell 8: Trade Analysis
    nb.cells.append(
        nbf.v4.new_markdown_cell("## 7. 거래 분석\n\n개별 거래의 수익/손실을 분석합니다.")
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Analyze trades\ntrades = results['trades']\ntrade_returns = [t['return'] for t in trades]\n\nfig, axes = plt.subplots(2, 2, figsize=(14, 10))\n\n# 1. Profit Distribution\naxes[0, 0].hist(trade_returns, bins=30, edgecolor='black', alpha=0.7)\naxes[0, 0].set_title('거래 수익 분포')\naxes[0, 0].set_xlabel('수익률')\naxes[0, 0].set_ylabel('빈도')\n\n# 2. Cumulative P&L\ncum_pnl = np.cumsum([t['pnl'] for t in trades])\naxes[0, 1].plot(cum_pnl, linewidth=2)\naxes[0, 1].set_title('누적 손익')\naxes[0, 1].set_xlabel('거래 번호')\naxes[0, 1].set_ylabel('손익 (원)')\naxes[0, 1].grid(True, alpha=0.3)\n\n# 3. Win/Loss Distribution\nwin_losses = pd.Series(trade_returns).apply(lambda x: 'Win' if x > 0 else 'Loss')\nwin_losses.value_counts().plot(kind='bar', ax=axes[1, 0], color=['green', 'red'])\naxes[1, 0].set_title('승리/손실 거래 수')\naxes[1, 0].set_ylabel('거래 수')\n\n# 4. Drawdown\nequity = results['equity']\nrunning_max = np.maximum.accumulate(equity)\ndrawdown = (equity - running_max) / running_max\naxes[1, 1].fill_between(results['dates'], drawdown, alpha=0.5, color='red')\naxes[1, 1].set_title('낙폭')\naxes[1, 1].set_ylabel('낙폭')\naxes[1, 1].grid(True, alpha=0.3)\n\nplt.tight_layout()\nplt.show()\n\nprint('✓ 거래 분석 완료')"
        )
    )

    # Cell 9: Conclusion
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 8. 결론 및 개선 사항\n\n"
            "### 주요 발견사항:\n\n"
            "1. **성과**: 변동성 돌파 전략은 일정한 수준의 수익성을 보여주었습니다.\n"
            "2. **위험**: 최대 낙폭은 관리 가능한 수준으로 유지되었습니다.\n"
            "3. **효율성**: 샤프 비율은 전략의 위험-조정 수익성을 나타냅니다.\n\n"
            "### 개선 사항:\n\n"
            "1. **파라미터 최적화**: 룩백 기간과 변동성 임계값을 최적화합니다.\n"
            "2. **리스크 관리**: Stop-loss와 position sizing을 추가합니다.\n"
            "3. **필터링**: 시장 변동성이 낮은 기간을 필터링합니다.\n"
            "4. **다중 자산**: 다양한 암호화폐에 적용합니다.\n\n"
            "### 실제 운영 전 체크리스트:\n\n"
            "- [ ] Out-of-sample 테스트 완료\n"
            "- [ ] Walk-forward 분석 수행\n"
            "- [ ] Paper trading으로 검증\n"
            "- [ ] 거래소 API 연동 테스트\n"
            "- [ ] 리스크 관리 규칙 설정\n"
            "- [ ] 모니터링 시스템 구축"
        )
    )

    return nb


def create_portfolio_optimization_notebook():
    """Create 02-Portfolio-Optimization.ipynb"""
    nb = nbf.v4.new_notebook()

    # Cell 1: Title
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "# 포트폴리오 최적화: 세 가지 방법론 비교\n\n"
            "이 노트북은 MPT, Risk Parity, Kelly Criterion 세 가지 포트폴리오 최적화 방법을 비교합니다."
        )
    )

    # Cell 2: Introduction
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. 포트폴리오 최적화 개요\n\n"
            "포트폴리오 최적화는 자산 배분을 결정하는 핵심 과제입니다.\n\n"
            "### 비교할 방법론:\n"
            "1. **Modern Portfolio Theory (MPT)**: Sharpe ratio 최대화\n"
            "2. **Risk Parity**: 각 자산의 위험 기여도 균등\n"
            "3. **Kelly Criterion**: 기대 수익에 따른 동적 배분"
        )
    )

    # Cell 3: Setup
    nb.cells.append(
        nbf.v4.new_code_cell(
            "import sys\nfrom pathlib import Path\n\n"
            "# Add project root to path\nproject_root = Path.cwd()\nif str(project_root) not in sys.path:\n    sys.path.insert(0, str(project_root))\n\n"
            "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom scipy.optimize import minimize\nimport warnings\nwarnings.filterwarnings('ignore')\n\n"
            "# Import from crypto-quant-system\nfrom src.risk.portfolio_optimization import (\n    optimize_portfolio_mpt,\n    calculate_risk_parity_weights,\n    optimize_kelly_criterion\n)\nfrom src.risk.metrics import calculate_metrics\n\n"
            "# Set up visualization\nplt.style.use('seaborn-v0_8-darkgrid')\nsns.set_palette('husl')\n\nprint('✓ 환경 설정 완료')"
        )
    )

    # Cell 4: Data Preparation
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. 데이터 준비\n\n포트폴리오에 포함할 자산의 수익률 데이터를 준비합니다."
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Example: Create sample returns for demonstration\n# In practice, you would load real market data\n\n"
            "# Assets: BTC, ETH, SOL, ADA\nassets = ['BTC', 'ETH', 'SOL', 'ADA']\nn_assets = len(assets)\nn_periods = 252  # 1 year of daily data\n\n"
            "# Generate correlated returns\nnp.random.seed(42)\nreturns_data = pd.DataFrame(\n    np.random.multivariate_normal(\n        [0.0005, 0.0004, 0.0003, 0.0002],\n        [[0.0004, 0.0002, 0.0001, 0.0001],\n         [0.0002, 0.0003, 0.0001, 0.00005],\n         [0.0001, 0.0001, 0.0005, 0.00008],\n         [0.0001, 0.00005, 0.00008, 0.0002]],\n        n_periods\n    ),\n    columns=assets\n)\n\n# Calculate statistics\ncov_matrix = returns_data.cov()\ncorr_matrix = returns_data.corr()\nmean_returns = returns_data.mean()\nstd_returns = returns_data.std()\n\nprint(f'포트폴리오 자산: {assets}')\nprint(f'\\n기대 연간 수익률:')\nprint((mean_returns * 252).to_string())\nprint(f'\\n연간 변동성:')\nprint((std_returns * np.sqrt(252)).to_string())"
        )
    )

    # Cell 5: MPT
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Modern Portfolio Theory (MPT)\n\n"
            "Sharpe ratio를 최대화하는 최적 포트폴리오를 찾습니다."
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Optimize for maximum Sharpe ratio\nweights_mpt = optimize_portfolio_mpt(\n    mean_returns=mean_returns,\n    cov_matrix=cov_matrix,\n    risk_free_rate=0.03\n)\n\n# Calculate portfolio metrics\nportfolio_return_mpt = np.sum(mean_returns * weights_mpt) * 252\nportfolio_vol_mpt = np.sqrt(np.dot(weights_mpt, np.dot(cov_matrix, weights_mpt))) * np.sqrt(252)\nsharpe_ratio_mpt = (portfolio_return_mpt - 0.03) / portfolio_vol_mpt\n\nprint('Modern Portfolio Theory (MPT)')\nprint('='*50)\nfor asset, weight in zip(assets, weights_mpt):\n    print(f'{asset:6s}: {weight:7.2%}')\nprint(f'\\n포트폴리오 수익률: {portfolio_return_mpt:.2%}')\nprint(f'포트폴리오 변동성: {portfolio_vol_mpt:.2%}')\nprint(f'Sharpe Ratio: {sharpe_ratio_mpt:.2f}')"
        )
    )

    # Cell 6: Risk Parity
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. Risk Parity\n\n각 자산의 위험 기여도를 균등하게 하는 방식입니다."
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Calculate Risk Parity weights\nweights_rp = calculate_risk_parity_weights(cov_matrix)\n\n# Calculate portfolio metrics\nportfolio_return_rp = np.sum(mean_returns * weights_rp) * 252\nportfolio_vol_rp = np.sqrt(np.dot(weights_rp, np.dot(cov_matrix, weights_rp))) * np.sqrt(252)\nsharpe_ratio_rp = (portfolio_return_rp - 0.03) / portfolio_vol_rp\n\nprint('Risk Parity')\nprint('='*50)\nfor asset, weight in zip(assets, weights_rp):\n    print(f'{asset:6s}: {weight:7.2%}')\nprint(f'\\n포트폴리오 수익률: {portfolio_return_rp:.2%}')\nprint(f'포트폴리오 변동성: {portfolio_vol_rp:.2%}')\nprint(f'Sharpe Ratio: {sharpe_ratio_rp:.2f}')"
        )
    )

    # Cell 7: Kelly Criterion
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Kelly Criterion\n\n장기 성장을 최대화하는 동적 배분 방식입니다."
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Calculate Kelly weights\n# Kelly Criterion: f* = (p*b - q) / b\n# Where p = win probability, q = loss probability, b = odds\n\nweights_kelly = optimize_kelly_criterion(\n    mean_returns=mean_returns,\n    cov_matrix=cov_matrix\n)\n\n# Calculate portfolio metrics\nportfolio_return_kelly = np.sum(mean_returns * weights_kelly) * 252\nportfolio_vol_kelly = np.sqrt(np.dot(weights_kelly, np.dot(cov_matrix, weights_kelly))) * np.sqrt(252)\nsharpe_ratio_kelly = (portfolio_return_kelly - 0.03) / portfolio_vol_kelly\n\nprint('Kelly Criterion')\nprint('='*50)\nfor asset, weight in zip(assets, weights_kelly):\n    print(f'{asset:6s}: {weight:7.2%}')\nprint(f'\\n포트폴리오 수익률: {portfolio_return_kelly:.2%}')\nprint(f'포트폴리오 변동성: {portfolio_vol_kelly:.2%}')\nprint(f'Sharpe Ratio: {sharpe_ratio_kelly:.2f}')"
        )
    )

    # Cell 8: Comparison
    nb.cells.append(nbf.v4.new_markdown_cell("## 6. 세 가지 방법론 비교"))

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Create comparison table\ncomparison_data = {\n    '방법론': ['MPT', 'Risk Parity', 'Kelly Criterion'],\n    '수익률': [f'{portfolio_return_mpt:.2%}', f'{portfolio_return_rp:.2%}', f'{portfolio_return_kelly:.2%}'],\n    '변동성': [f'{portfolio_vol_mpt:.2%}', f'{portfolio_vol_rp:.2%}', f'{portfolio_vol_kelly:.2%}'],\n    'Sharpe Ratio': [f'{sharpe_ratio_mpt:.2f}', f'{sharpe_ratio_rp:.2f}', f'{sharpe_ratio_kelly:.2f}']\n}\n\ncomparison_df = pd.DataFrame(comparison_data)\nprint(comparison_df.to_string(index=False))\n\n# Visualization\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n# Weights comparison\nweights_df = pd.DataFrame({\n    'MPT': weights_mpt,\n    'Risk Parity': weights_rp,\n    'Kelly': weights_kelly\n}, index=assets)\n\nweights_df.plot(kind='bar', ax=axes[0])\naxes[0].set_title('포트폴리오 구성 비교')\naxes[0].set_ylabel('가중치')\naxes[0].legend()\naxes[0].grid(True, alpha=0.3)\n\n# Efficient Frontier\nportfolios = np.random.multivariate_normal(\n    mean_returns, cov_matrix, 1000\n)\nreturns_plot = []\nvols_plot = []\nfor _ in range(1000):\n    w = np.random.dirichlet(np.ones(n_assets))\n    returns_plot.append(np.sum(mean_returns * w) * 252)\n    vols_plot.append(np.sqrt(np.dot(w, np.dot(cov_matrix, w))) * np.sqrt(252))\n\naxes[1].scatter(vols_plot, returns_plot, alpha=0.3, s=10)\naxes[1].scatter(portfolio_vol_mpt, portfolio_return_mpt, marker='*', s=500, label='MPT', color='red')\naxes[1].scatter(portfolio_vol_rp, portfolio_return_rp, marker='s', s=200, label='Risk Parity', color='green')\naxes[1].scatter(portfolio_vol_kelly, portfolio_return_kelly, marker='^', s=200, label='Kelly', color='blue')\naxes[1].set_xlabel('변동성')\naxes[1].set_ylabel('기대 수익률')\naxes[1].set_title('효율적 프론티어')\naxes[1].legend()\naxes[1].grid(True, alpha=0.3)\n\nplt.tight_layout()\nplt.show()\n\nprint('✓ 비교 분석 완료')"
        )
    )

    # Cell 9: Transaction Costs
    nb.cells.append(nbf.v4.new_markdown_cell("## 7. 거래 비용 고려"))

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Consider transaction costs\ntransaction_cost_rate = 0.0005  # 0.05%\n\n# Rebalancing frequency: quarterly\nrebalancing_per_year = 4\nannual_transaction_cost_mpt = (weights_mpt.sum() / 2 * transaction_cost_rate * rebalancing_per_year)\nannual_transaction_cost_rp = (weights_rp.sum() / 2 * transaction_cost_rate * rebalancing_per_year)\nannual_transaction_cost_kelly = (weights_kelly.sum() / 2 * transaction_cost_rate * rebalancing_per_year)\n\n# Net returns after costs\nnet_return_mpt = portfolio_return_mpt - annual_transaction_cost_mpt\nnet_return_rp = portfolio_return_rp - annual_transaction_cost_rp\nnet_return_kelly = portfolio_return_kelly - annual_transaction_cost_kelly\n\nprint('거래 비용 고려 후 수익률')\nprint('='*50)\nprint(f'MPT: {portfolio_return_mpt:.2%} - {annual_transaction_cost_mpt:.2%} = {net_return_mpt:.2%}')\nprint(f'Risk Parity: {portfolio_return_rp:.2%} - {annual_transaction_cost_rp:.2%} = {net_return_rp:.2%}')\nprint(f'Kelly: {portfolio_return_kelly:.2%} - {annual_transaction_cost_kelly:.2%} = {net_return_kelly:.2%}')"
        )
    )

    # Cell 10: Conclusion
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 8. 결론\n\n"
            "### 각 방법론의 특징:\n\n"
            "**MPT (Modern Portfolio Theory)**\n"
            "- 장점: Sharpe ratio 최적화, 수학적으로 엄밀함\n"
            "- 단점: 과거 데이터에 의존, 극단적인 가중치 가능\n\n"
            "**Risk Parity**\n"
            "- 장점: 균형잡힌 위험 배분, 구현이 간단\n"
            "- 단점: 수익률 목표가 없음, 상관관계 변화에 취약\n\n"
            "**Kelly Criterion**\n"
            "- 장점: 장기 성장 최대화, 동적 적응\n"
            "- 단점: 변동성이 높을 수 있음, 추정치에 민감\n\n"
            "### 실무 권장사항:\n\n"
            "1. **포트폴리오 규모**: 작을수록 Kelly, 클수록 MPT\n"
            "2. **리스크 선호도**: 보수적이면 Risk Parity, 공격적이면 Kelly\n"
            "3. **운영 환경**: 복잡할수록 MPT, 간단할수록 Risk Parity\n"
            "4. **하이브리드 접근**: 여러 방법을 결합하여 사용"
        )
    )

    return nb


def create_live_trading_notebook():
    """Create 03-Live-Trading-Analysis.ipynb"""
    nb = nbf.v4.new_notebook()

    # Cell 1: Title
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "# 실시간 거래 분석: 시뮬레이션 및 리스크 관리\n\n"
            "이 노트북은 실시간 거래 환경을 시뮬레이션하고 리스크 관리 메커니즘을 검증합니다."
        )
    )

    # Cell 2: Setup
    nb.cells.append(
        nbf.v4.new_code_cell(
            "import sys\nfrom pathlib import Path\n\n"
            "# Add project root to path\nproject_root = Path.cwd()\nif str(project_root) not in sys.path:\n    sys.path.insert(0, str(project_root))\n\n"
            "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom datetime import datetime, timedelta\nimport warnings\nwarnings.filterwarnings('ignore')\n\n"
            "# Import from crypto-quant-system\nfrom src.execution.bot_facade import BotFacade\nfrom src.risk.metrics import calculate_var, calculate_cvar, calculate_sharpe\nfrom src.strategies.momentum import SimpleMomentumStrategy\n\n"
            "# Set up visualization\nplt.style.use('seaborn-v0_8-darkgrid')\nsns.set_palette('husl')\n\nprint('✓ 환경 설정 완료')"
        )
    )

    # Cell 3: Price Simulation
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. 가격 시뮬레이션 (Geometric Brownian Motion)\n\n"
            "실시간 거래 환경을 시뮬레이션하기 위해 GBM 모델을 사용합니다."
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# GBM Price Simulator\nclass PriceSimulator:\n    def __init__(self, initial_price, drift=0.0005, volatility=0.02):\n        self.price = initial_price\n        self.drift = drift\n        self.volatility = volatility\n        self.prices = [initial_price]\n        self.timestamps = [0]\n        \n    def simulate(self, time_steps=100, dt=1.0):\n        '''Simulate price path using GBM'''\n        for t in range(1, time_steps + 1):\n            dW = np.random.normal(0, np.sqrt(dt))\n            dP = self.drift * self.price * dt + self.volatility * self.price * dW\n            self.price += dP\n            self.prices.append(max(self.price, 0))  # Prevent negative prices\n            self.timestamps.append(t)\n        return self.prices\n\n# Initialize simulator\nsimulator = PriceSimulator(initial_price=50000, drift=0.0005, volatility=0.02)\nprices = simulator.simulate(time_steps=250)  # 250 trading days\n\nprint(f'초기 가격: {prices[0]:.2f}')\nprint(f'최종 가격: {prices[-1]:.2f}')\nprint(f'최고 가격: {max(prices):.2f}')\nprint(f'최저 가격: {min(prices):.2f}')"
        )
    )

    # Cell 4: Portfolio & Strategy
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. 포트폴리오 및 전략 설정"))

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Simple Portfolio Class\nclass Portfolio:\n    def __init__(self, initial_capital=1000000):\n        self.cash = initial_capital\n        self.positions = {}\n        self.equity_history = [initial_capital]\n        self.trades = []\n        \n    def buy(self, asset, price, quantity):\n        '''Buy asset'''\n        cost = price * quantity\n        if cost <= self.cash:\n            self.cash -= cost\n            if asset not in self.positions:\n                self.positions[asset] = 0\n            self.positions[asset] += quantity\n            self.trades.append(('BUY', asset, price, quantity))\n            return True\n        return False\n    \n    def sell(self, asset, price, quantity):\n        '''Sell asset'''\n        if asset in self.positions and self.positions[asset] >= quantity:\n            self.cash += price * quantity\n            self.positions[asset] -= quantity\n            self.trades.append(('SELL', asset, price, quantity))\n            return True\n        return False\n    \n    def get_total_value(self, current_price, asset='BTC/KRW'):\n        '''Get total portfolio value'''\n        asset_value = self.positions.get(asset, 0) * current_price\n        return self.cash + asset_value\n    \n    def calculate_metrics(self, prices):\n        '''Calculate portfolio metrics'''\n        values = []\n        for price in prices:\n            value = self.get_total_value(price)\n            values.append(value)\n        \n        returns = np.diff(values) / np.array(values[:-1])\n        \n        # VaR (95% confidence)\n        var_95 = np.percentile(returns, 5)\n        \n        # CVaR (Conditional VaR)\n        cvar_95 = returns[returns <= var_95].mean()\n        \n        # Sharpe Ratio\n        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)\n        \n        # Maximum Drawdown\n        running_max = np.maximum.accumulate(values)\n        drawdown = (values - running_max) / running_max\n        max_dd = np.min(drawdown)\n        \n        return {\n            'values': values,\n            'var_95': var_95,\n            'cvar_95': cvar_95,\n            'sharpe': sharpe,\n            'max_dd': max_dd\n        }\n\n# Initialize portfolio\nportfolio = Portfolio(initial_capital=1000000)\n\nprint('포트폴리오 초기화 완료')\nprint(f'초기 자본: {portfolio.get_total_value(prices[0]):,.0f} 원')"
        )
    )

    # Cell 5: Trading Simulation
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. 실시간 거래 시뮬레이션"))

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Simple Momentum Strategy\nclass SimpleMomentumStrategy:\n    def __init__(self, lookback=20, threshold=0.02):\n        self.lookback = lookback\n        self.threshold = threshold\n        self.price_history = []\n        \n    def generate_signal(self, price):\n        '''Generate trading signal'''\n        self.price_history.append(price)\n        \n        if len(self.price_history) < self.lookback:\n            return None\n        \n        recent_prices = self.price_history[-self.lookback:]\n        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]\n        \n        if momentum > self.threshold:\n            return 'BUY'\n        elif momentum < -self.threshold:\n            return 'SELL'\n        return None\n\n# Initialize strategy\nstrategy = SimpleMomentumStrategy(lookback=20, threshold=0.02)\n\n# Simulation\nfor i, price in enumerate(prices):\n    signal = strategy.generate_signal(price)\n    \n    if signal == 'BUY' and portfolio.cash > price * 10:\n        portfolio.buy('BTC/KRW', price, 1)\n    elif signal == 'SELL' and portfolio.positions.get('BTC/KRW', 0) > 0:\n        portfolio.sell('BTC/KRW', price, 1)\n    \n    portfolio.equity_history.append(portfolio.get_total_value(price))\n\nprint(f'총 거래 수: {len(portfolio.trades)}')\nprint(f'최종 자산: {portfolio.equity_history[-1]:,.0f} 원')\nprint(f'수익: {(portfolio.equity_history[-1] / portfolio.equity_history[0] - 1):.2%}')"
        )
    )

    # Cell 6: Risk Metrics
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. 리스크 메트릭 계산"))

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Calculate metrics\nmetrics = portfolio.calculate_metrics(prices)\n\nprint('포트폴리오 리스크 메트릭')\nprint('='*50)\nprint(f'Value at Risk (95%): {metrics[\"var_95\"]:.4f}')\nprint(f'Conditional VaR (95%): {metrics[\"cvar_95\"]:.4f}')\nprint(f'Sharpe Ratio: {metrics[\"sharpe\"]:.2f}')\nprint(f'Maximum Drawdown: {metrics[\"max_dd\"]:.2%}')\nprint(f'\\n최종 자산: {metrics[\"values\"][-1]:,.0f} 원')\nprint(f'초기 자산: {metrics[\"values\"][0]:,.0f} 원')\nprint(f'수익률: {(metrics[\"values\"][-1] / metrics[\"values\"][0] - 1):.2%}')"
        )
    )

    # Cell 7: Visualization
    nb.cells.append(nbf.v4.new_markdown_cell("## 5. 거래 결과 시각화"))

    nb.cells.append(
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n\n# 1. Portfolio Value\naxes[0, 0].plot(metrics['values'], linewidth=2, label='Portfolio Value')\naxes[0, 0].fill_between(range(len(metrics['values'])), metrics['values'], alpha=0.3)\naxes[0, 0].set_title('포트폴리오 가치')\naxes[0, 0].set_ylabel('가치 (원)')\naxes[0, 0].legend()\naxes[0, 0].grid(True, alpha=0.3)\n\n# 2. Price vs Portfolio\nax2_1 = axes[0, 1]\nax2_1_twin = ax2_1.twinx()\nax2_1.plot(prices, color='blue', linewidth=2, label='BTC Price')\nax2_1_twin.plot(metrics['values'], color='red', linewidth=2, label='Portfolio Value')\nax2_1.set_title('가격 vs 포트폴리오 가치')\nax2_1.set_ylabel('가격 (원)', color='blue')\nax2_1_twin.set_ylabel('포트폴리오 (원)', color='red')\nax2_1.grid(True, alpha=0.3)\n\n# 3. Drawdown\nrunning_max = np.maximum.accumulate(metrics['values'])\ndrawdown = (np.array(metrics['values']) - running_max) / running_max\naxes[1, 0].fill_between(range(len(drawdown)), drawdown, alpha=0.5, color='red')\naxes[1, 0].set_title('낙폭 (Drawdown)')\naxes[1, 0].set_ylabel('낙폭')\naxes[1, 0].grid(True, alpha=0.3)\n\n# 4. Returns Distribution\nreturns = np.diff(metrics['values']) / np.array(metrics['values'][:-1])\naxes[1, 1].hist(returns, bins=30, edgecolor='black', alpha=0.7)\naxes[1, 1].axvline(metrics['var_95'], color='red', linestyle='--', label=f'VaR 95% ({metrics[\"var_95\"]:.4f})')\naxes[1, 1].set_title('수익률 분포')\naxes[1, 1].set_xlabel('수익률')\naxes[1, 1].legend()\naxes[1, 1].grid(True, alpha=0.3)\n\nplt.tight_layout()\nplt.show()\n\nprint('✓ 시각화 완료')"
        )
    )

    # Cell 8: Risk Management Checklist
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 6. 실시간 거래 전 체크리스트\n\n"
            "실제 거래 실행 전에 다음 사항들을 확인하세요:\n\n"
            "### 기술 검증\n"
            "- [ ] 백테스팅 완료 (최소 1년 데이터)\n"
            "- [ ] Walk-forward 분석 수행\n"
            "- [ ] 거래소 API 연동 테스트\n"
            "- [ ] Paper trading으로 1개월 이상 검증\n"
            "- [ ] 거래 지연 및 슬리피지 테스트\n\n"
            "### 리스크 관리\n"
            "- [ ] 일일 손실 한도 설정 (예: 초기 자본의 2%)\n"
            "- [ ] 포지션 크기 제한 (예: 한 거래당 초기 자본의 1%)\n"
            "- [ ] Stop-loss 규칙 설정 (예: -5% 손실)\n"
            "- [ ] Take-profit 규칙 설정 (예: +10% 수익)\n"
            "- [ ] 포트폴리오 리밸런싱 주기 설정\n"
            "- [ ] VaR 및 CVaR 모니터링\n\n"
            "### 운영 체계\n"
            "- [ ] 거래 로깅 시스템 구축\n"
            "- [ ] 실시간 모니터링 대시보드\n"
            "- [ ] 긴급 정지(Kill switch) 메커니즘\n"
            "- [ ] 텔레그램/이메일 알림 설정\n"
            "- [ ] 일일 리포팅 자동화\n"
            "- [ ] 주간 성과 검토 프로세스\n\n"
            "### 거래소 설정\n"
            "- [ ] API 키 안전 저장 (환경 변수 사용)\n"
            "- [ ] 거래소별 제한사항 확인\n"
            "- [ ] 최소 주문량 확인\n"
            "- [ ] 수수료율 확인\n"
            "- [ ] 입출금 제한 확인\n\n"
            "### 자본 관리\n"
            "- [ ] 초기 투자 규모 결정\n"
            "- [ ] 손익 분리 규칙 설정\n"
            "- [ ] 자본 추가/인출 정책\n"
            "- [ ] 세금 계산 방법 결정\n"
            "- [ ] 거래 기록 보관"
        )
    )

    # Cell 9: Conclusion
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 7. 결론\n\n"
            "### 핵심 학습 사항:\n\n"
            "1. **시뮬레이션의 중요성**: 실제 거래 전 충분한 시뮬레이션이 필수\n"
            "2. **리스크 관리**: VaR, CVaR 등 정량적 지표로 리스크 측정\n"
            "3. **지속적 모니터링**: 거래 수행 중에도 메트릭 추적\n"
            "4. **감정 제어**: 자동화된 규칙 따르기\n"
            "5. **성능 분석**: 정기적인 거래 검토 및 개선\n\n"
            "### 다음 단계:\n\n"
            "1. 실제 거래소에서 Paper Trading 실행\n"
            "2. 소규모로 실제 거래 시작 (자본의 10% 이하)\n"
            "3. 성과 모니터링 및 전략 개선\n"
            "4. 점진적으로 거래량 증가\n"
            "5. 새로운 자산/전략 추가\n\n"
            "⚠️ **주의**: 암호화폐 거래는 높은 위험을 수반합니다. "
            "여유 자금으로만 거래하고, 충분한 학습과 테스트 후에 시작하세요."
        )
    )

    return nb


# Create all notebooks
print("노트북 생성 중...")
notebooks = {
    "notebooks/01-Backtesting-Case-Study.ipynb": create_backtesting_notebook(),
    "notebooks/02-Portfolio-Optimization.ipynb": create_portfolio_optimization_notebook(),
    "notebooks/03-Live-Trading-Analysis.ipynb": create_live_trading_notebook(),
}

# Save notebooks
for path, nb in notebooks.items():
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"OK {path} 생성 완료")

print("\n✓ 모든 노트북이 정상적으로 생성되었습니다!")
