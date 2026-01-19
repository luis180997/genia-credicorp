"""
Genia AI Agent - Herramientas
Herramientas disponibles para el agente de Credicorp Capital.

Herramientas:
1. search_internal_documents: Busca en documentos internos (RAG)
2. search_financial_web: Busca noticias/información financiera en la web (DuckDuckGo)
3. get_stock_info: Consulta datos de acciones (Yahoo Finance)
4. get_exchange_rate: Consulta tipos de cambio (relevante para USD/PEN)
5. financial_calculator: Cálculos financieros básicos
"""
from typing import List
from langchain.tools import tool
try:
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:
    pass  # Allow import even if duckduckgo is missing (will fail at runtime if used)

try:
    import yfinance as yf
except ImportError:
    pass  # Allow import even if yfinance is missing

from ..config import settings


def create_document_search_tool(vector_store_service):
    """
    Crea herramienta para buscar en documentos internos de Credicorp.
    PRIORIDAD 1: Usar primero para preguntas sobre políticas y procedimientos.
    """
    
    @tool
    def search_internal_documents(query: str) -> str:
        """
        Busca información en los documentos internos de Credicorp Capital.
        
        SIEMPRE usa esta herramienta PRIMERO cuando el usuario pregunte sobre:
        - Políticas de inversión de Credicorp
        - Procedimientos internos
        - Reportes y documentos corporativos
        - Información específica de productos de Credicorp
        
        Args:
            query: La pregunta o términos de búsqueda
            
        Returns:
            Fragmentos relevantes de documentos internos con sus fuentes
        """
        # Usar el valor configurado para K
        results = vector_store_service.search(query, k=settings.search_k_retrieval)
        
        if not results:
            return (
                "No encontré documentos internos relevantes para esta consulta. "
                "Puedes usar otras herramientas para buscar información externa."
            )
        
        formatted_results = []
        for doc, score in results:
            source = doc.metadata.get('source', 'Documento desconocido')
            doc_type = doc.metadata.get('document_type', 'general')
            content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            
            formatted_results.append(
                f"Fuente: {source} (Tipo: {doc_type})\n"
                f"Contenido: {content}\n"
            )
        
        return "\n---\n".join(formatted_results)
    
    return search_internal_documents


def create_web_search_tool():
    """
    Crea herramienta para búsqueda web de noticias e información financiera.
    Usa DuckDuckGo como motor de búsqueda.
    """
    
    @tool
    def search_financial_web(query: str) -> str:
        """
        Busca información financiera actualizada en internet.
        
        Usa esta herramienta para:
        - Noticias recientes del mercado
        - Información sobre empresas públicas
        - Tendencias económicas
        - Eventos financieros actuales
        
        Args:
            query: Términos de búsqueda (añade contexto financiero si es relevante)
            
        Returns:
            Resultados de búsqueda web
        """
        try:
            search = DuckDuckGoSearchRun()
            
            # Añadir contexto financiero
            enhanced_query = f"{query} finanzas inversión mercado"
            results = search.run(enhanced_query)
            
            return f"Resultados de búsqueda web:\n\n{results}"
        except ImportError:
             return "Error: duckduckgo-search no está instalada."
        except Exception as e:
            return f"Error en búsqueda web: {str(e)}"
    
    return search_financial_web


def create_stock_info_tool():
    """
    Crea herramienta para consultar datos de acciones via Yahoo Finance.
    """
    
    @tool
    def get_stock_info(ticker: str) -> str:
        """
        Obtiene información de mercado para una acción, ETF o índice.
        
        Usa esta herramienta para:
        - Cotizaciones actuales de acciones (ej: AAPL, MSFT, BVN)
        - Información de empresas públicas
        - Precios de ETFs (ej: SPY, QQQ)
        - Datos de acciones peruanas (ej: BVN, CREDICORPI.LM)
        
        Args:
            ticker: Símbolo bursátil (ej: AAPL, MSFT, BVN)
            
        Returns:
            Información de mercado del activo
        """
        try:
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            
            if not info or 'shortName' not in info:
                return f"No encontré información para '{ticker}'. Verifica el símbolo."
            
            name = info.get('shortName', ticker)
            price = info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')
            currency = info.get('currency', 'USD')
            change = info.get('regularMarketChangePercent', 0)
            volume = info.get('volume', 0)
            market_cap = info.get('marketCap', 0)
            sector = info.get('sector', 'N/A')
            
            # Formatear market cap
            if market_cap >= 1e12:
                mc_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                mc_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                mc_str = f"${market_cap/1e6:.2f}M"
            else:
                mc_str = f"${market_cap:,.0f}"
            
            return f"""
{name} ({ticker.upper()})

- Precio: {currency} {price:,.2f}
- Cambio: {change:+.2f}%
- Volumen: {volume:,}
- Cap. Mercado: {mc_str}
- Sector: {sector}

Fuente: Yahoo Finance (puede tener retraso de 15-20 min)
"""
        except ImportError:
            return "yfinance no instalado. Usa búsqueda web para cotizaciones."
        except NameError:
             return "yfinance no instalado o no importado correctamente."
        except Exception as e:
            return f"Error al consultar {ticker}: {str(e)}"
    
    return get_stock_info


def create_exchange_rate_tool():
    """
    Crea herramienta para consultar tipos de cambio.
    Muy relevante para Credicorp (operaciones en USD y PEN).
    """
    
    @tool
    def get_exchange_rate(from_currency: str, to_currency: str = "PEN") -> str:
        """
        Obtiene el tipo de cambio entre dos monedas.
        
        Muy útil para:
        - Tipo de cambio USD/PEN (dólar a sol peruano)
        - Conversiones EUR/USD
        - Cualquier par de divisas
        
        Args:
            from_currency: Moneda origen (ej: USD, EUR)
            to_currency: Moneda destino (ej: PEN, USD). Default: PEN
            
        Returns:
            Tipo de cambio actual
        """
        try:
            # Yahoo Finance usa formato USDPEN=X para forex
            pair = f"{from_currency.upper()}{to_currency.upper()}=X"
            ticker = yf.Ticker(pair)
            
            # Obtener último precio
            hist = ticker.history(period="1d")
            if hist.empty:
                return f"No encontré tipo de cambio para {from_currency}/{to_currency}"
            
            rate = hist['Close'].iloc[-1]
            
            return f"""
Tipo de Cambio {from_currency.upper()}/{to_currency.upper()}

- Tasa actual: {rate:.4f}
- 1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()}

Ejemplo: 100 {from_currency.upper()} = {100 * rate:,.2f} {to_currency.upper()}

Fuente: Yahoo Finance
"""
        except ImportError:
            return "yfinance no instalado."
        except NameError:
             return "yfinance no instalado o no importado correctamente."
        except Exception as e:
            return f"Error: {str(e)}"
    
    return get_exchange_rate


def create_calculator_tool():
    """
    Herramienta para cálculos financieros básicos.
    """
    
    @tool
    def financial_calculator(expression: str) -> str:
        """
        Realiza cálculos matemáticos y financieros.
        
        Usa para:
        - Calcular rendimientos: "100000 * 0.05"
        - Interés compuesto: "10000 * (1 + 0.08) ** 5"
        - Conversiones: "1000 * 3.75" (USD a PEN)
        
        Args:
            expression: Expresión matemática (usa ** para potencias)
            
        Returns:
            Resultado del cálculo
        """
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "Error: Solo se permiten números y operadores (+, -, *, /, **)"
            
            result = eval(expression)
            
            if isinstance(result, float):
                if result >= 1e6:
                    return f"Resultado: {result:,.2f} ({result/1e6:.2f} millones)"
                return f"Resultado: {result:,.2f}"
            return f"Resultado: {result:,}"
            
        except ZeroDivisionError:
            return "Error: División por cero"
        except Exception as e:
            return f"Error: {str(e)}"
    
    return financial_calculator


def get_all_tools(vector_store_service) -> List:
    """
    Retorna todas las herramientas disponibles para el agente Genia.
    
    Herramientas ordenadas por prioridad de uso:
    1. Documentos internos (siempre primero para info de Credicorp)
    2. Yahoo Finance (cotizaciones precisas)
    3. Tipos de cambio (relevante para operaciones Perú)
    4. Búsqueda web (noticias y info general)
    5. Calculadora (operaciones numéricas)
    """
    return [
        create_document_search_tool(vector_store_service),
        create_stock_info_tool(),
        create_exchange_rate_tool(),
        create_web_search_tool(),
        create_calculator_tool(),
    ]
