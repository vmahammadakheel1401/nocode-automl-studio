import { buildDatasetFromCSV } from './csvParser';
import { ParsedDataset } from './types';

function createRng(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function normalNoise(rng: () => number, std = 1): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z * std;
}

export function generateHousePricesCSV(): string {
  const rng = createRng(4201);
  const neighborhoods = [
    { name: 'Downtown', premium: 95000 },
    { name: 'Oakridge', premium: 45000 },
    { name: 'Riverside', premium: 68000 },
    { name: 'Westside', premium: 25000 },
    { name: 'Hillcrest', premium: 120000 },
  ];
  const conditions = [
    { name: 'Fair', mult: 0.91 },
    { name: 'Good', mult: 1.0 },
    { name: 'Excellent', mult: 1.13 },
  ];

  const headers = [
    'property_id',
    'sqft_living',
    'bedrooms',
    'bathrooms',
    'lot_size_sqft',
    'year_built',
    'neighborhood',
    'condition',
    'garage_spaces',
    'sale_price',
  ];

  const lines = [headers.join(',')];

  for (let i = 1; i <= 180; i++) {
    const propertyId = `PROP-${String(1000 + i)}`;
    const sqft = Math.round(950 + rng() * 2650);
    const bedrooms = Math.min(6, Math.max(1, Math.round(sqft / 620 + normalNoise(rng, 0.45))));
    const bathrooms = Number(
      Math.min(4.5, Math.max(1, Math.round((bedrooms * 0.72 + rng() * 0.6) * 2) / 2)).toFixed(1)
    );
    const lotSize = Math.round(2800 + sqft * 1.9 + rng() * 4200);
    const yearBuilt = Math.round(1962 + rng() * 62);
    const nb = neighborhoods[Math.floor(rng() * neighborhoods.length)];
    const cond = conditions[Math.floor(rng() * conditions.length)];
    const garage = Math.min(3, Math.max(0, Math.round(sqft / 1200 + (rng() - 0.35))));

    // Realistic pricing formula with mild non-linearity
    const basePrice =
      115000 +
      sqft * 195 +
      bedrooms * 14500 +
      bathrooms * 22000 +
      lotSize * 6.5 +
      (yearBuilt - 1960) * 1650 +
      garage * 16000 +
      nb.premium;

    const salePrice = Math.round(
      (basePrice * cond.mult + normalNoise(rng, 18500)) / 500
    ) * 500;

    // Sprinkle a tiny amount (~1.5%) of realistic missing cells in non-target features
    const lotOut = rng() < 0.025 ? '' : String(lotSize);
    const garageOut = rng() < 0.02 ? '' : String(garage);
    const condOut = rng() < 0.015 ? '' : cond.name;

    lines.push(
      [
        propertyId,
        sqft,
        bedrooms,
        bathrooms,
        lotOut,
        yearBuilt,
        nb.name,
        condOut,
        garageOut,
        Math.max(145000, salePrice),
      ].join(',')
    );
  }

  return lines.join('\n');
}

export function generateCustomerChurnCSV(): string {
  const rng = createRng(8817);
  const contracts = ['Month-to-Month', 'One Year', 'Two Year'];
  const payments = ['Electronic Check', 'Credit Card', 'Bank Transfer'];
  const internetServices = ['Fiber Optic', 'DSL', 'None'];

  const headers = [
    'customer_id',
    'tenure_months',
    'monthly_charges',
    'support_tickets',
    'contract_type',
    'payment_method',
    'internet_service',
    'paperless_billing',
    'churn',
  ];

  const lines = [headers.join(',')];

  for (let i = 1; i <= 200; i++) {
    const customerId = `CUST-${String(5000 + i)}`;
    const contract = contracts[Math.floor(rng() * contracts.length)];
    const tenureBase =
      contract === 'Two Year' ? 44 : contract === 'One Year' ? 28 : 12;
    const tenure = Math.min(
      72,
      Math.max(1, Math.round(tenureBase + normalNoise(rng, 14)))
    );

    const internet =
      internetServices[Math.floor(rng() * internetServices.length)];
    const baseCharge =
      internet === 'Fiber Optic' ? 88 : internet === 'DSL' ? 56 : 28;
    const monthlyCharges = Number(
      Math.max(20, Math.min(125, baseCharge + normalNoise(rng, 13))).toFixed(2)
    );

    const payment = payments[Math.floor(rng() * payments.length)];
    const paperless = rng() < 0.62 ? 'Yes' : 'No';

    // Logit score driving churn
    let logit = -0.7;
    if (contract === 'Month-to-Month') logit += 1.45;
    else if (contract === 'Two Year') logit -= 1.35;
    else logit -= 0.45;

    logit -= (tenure - 24) * 0.045;
    logit += (monthlyCharges - 65) * 0.025;
    if (payment === 'Electronic Check') logit += 0.65;
    if (internet === 'Fiber Optic') logit += 0.45;
    if (paperless === 'Yes') logit += 0.25;

    const ticketLambda = logit > 0.2 ? 2.6 : 0.9;
    const supportTickets = Math.min(
      8,
      Math.max(0, Math.round(ticketLambda + Math.abs(normalNoise(rng, 1.1))))
    );
    logit += (supportTickets - 1.5) * 0.48;

    const prob = 1 / (1 + Math.exp(-logit));
    const churn = rng() < prob ? 'Yes' : 'No';

    // Sprinkle small realistic missing values
    const chargesOut = rng() < 0.02 ? '' : String(monthlyCharges);
    const paymentOut = rng() < 0.02 ? '' : payment;

    lines.push(
      [
        customerId,
        tenure,
        chargesOut,
        supportTickets,
        contract,
        paymentOut,
        internet,
        paperless,
        churn,
      ].join(',')
    );
  }

  return lines.join('\n');
}

export type SampleDatasetKey = 'house_prices' | 'customer_churn';

export const SAMPLE_DATASET_META: Record<
  SampleDatasetKey,
  {
    key: SampleDatasetKey;
    title: string;
    dropdownLabel: string;
    fileName: string;
    taskBadge: string;
    targetColumn: string;
    description: string;
    rowCountHint: number;
    colCountHint: number;
  }
> = {
  house_prices: {
    key: 'house_prices',
    title: 'House Prices',
    dropdownLabel: 'House Prices - Regression',
    fileName: 'house_prices.csv',
    taskBadge: 'Regression',
    targetColumn: 'sale_price',
    description:
      'Residential property listings with square footage, lot size, neighborhood, condition, and sale prices.',
    rowCountHint: 180,
    colCountHint: 10,
  },
  customer_churn: {
    key: 'customer_churn',
    title: 'Customer Churn',
    dropdownLabel: 'Customer Churn - Classification',
    fileName: 'customer_churn.csv',
    taskBadge: 'Classification',
    targetColumn: 'churn',
    description:
      'Subscription telemetry including tenure, contract terms, monthly charges, support tickets, and churn outcome.',
    rowCountHint: 200,
    colCountHint: 9,
  },
};

export function loadSampleDataset(key: SampleDatasetKey): ParsedDataset {
  const meta = SAMPLE_DATASET_META[key];
  const csv =
    key === 'house_prices'
      ? generateHousePricesCSV()
      : generateCustomerChurnCSV();
  return buildDatasetFromCSV(csv, meta.fileName, meta.title, meta.targetColumn);
}
