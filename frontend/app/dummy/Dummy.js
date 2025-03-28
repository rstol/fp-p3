'use strict';
var __awaiter =
  (this && this.__awaiter) ||
  function (thisArg, _arguments, P, generator) {
    function adopt(value) {
      return value instanceof P
        ? value
        : new P(function (resolve) {
            resolve(value);
          });
    }
    return new (P || (P = Promise))(function (resolve, reject) {
      function fulfilled(value) {
        try {
          step(generator.next(value));
        } catch (e) {
          reject(e);
        }
      }
      function rejected(value) {
        try {
          step(generator['throw'](value));
        } catch (e) {
          reject(e);
        }
      }
      function step(result) {
        result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected);
      }
      step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
  };
var __generator =
  (this && this.__generator) ||
  function (thisArg, body) {
    var _ = {
        label: 0,
        sent: function () {
          if (t[0] & 1) throw t[1];
          return t[1];
        },
        trys: [],
        ops: [],
      },
      f,
      y,
      t,
      g = Object.create((typeof Iterator === 'function' ? Iterator : Object).prototype);
    return (
      (g.next = verb(0)),
      (g['throw'] = verb(1)),
      (g['return'] = verb(2)),
      typeof Symbol === 'function' &&
        (g[Symbol.iterator] = function () {
          return this;
        }),
      g
    );
    function verb(n) {
      return function (v) {
        return step([n, v]);
      };
    }
    function step(op) {
      if (f) throw new TypeError('Generator is already executing.');
      while ((g && ((g = 0), op[0] && (_ = 0)), _))
        try {
          if (
            ((f = 1),
            y &&
              (t =
                op[0] & 2
                  ? y['return']
                  : op[0]
                    ? y['throw'] || ((t = y['return']) && t.call(y), 0)
                    : y.next) &&
              !(t = t.call(y, op[1])).done)
          )
            return t;
          if (((y = 0), t)) op = [op[0] & 2, t.value];
          switch (op[0]) {
            case 0:
            case 1:
              t = op;
              break;
            case 4:
              _.label++;
              return { value: op[1], done: false };
            case 5:
              _.label++;
              y = op[1];
              op = [0];
              continue;
            case 7:
              op = _.ops.pop();
              _.trys.pop();
              continue;
            default:
              if (
                !((t = _.trys), (t = t.length > 0 && t[t.length - 1])) &&
                (op[0] === 6 || op[0] === 2)
              ) {
                _ = 0;
                continue;
              }
              if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) {
                _.label = op[1];
                break;
              }
              if (op[0] === 6 && _.label < t[1]) {
                _.label = t[1];
                t = op;
                break;
              }
              if (t && _.label < t[2]) {
                _.label = t[2];
                _.ops.push(op);
                break;
              }
              if (t[2]) _.ops.pop();
              _.trys.pop();
              continue;
          }
          op = body.call(thisArg, _);
        } catch (e) {
          op = [6, e];
          y = 0;
        } finally {
          f = t = 0;
        }
      if (op[0] & 5) throw op[1];
      return { value: op[0] ? op[1] : void 0, done: true };
    }
  };
Object.defineProperty(exports, '__esModule', { value: true });
exports.Dummy = Dummy;
var react_1 = require('react');
var DataChoice_1 = require('~/components/DataChoice');
var ScatterPlot_1 = require('~/components/ScatterPlot');
var const_1 = require('~/lib/const');
function postPoints(choice) {
  return __awaiter(this, void 0, void 0, function () {
    var response, data, err_1;
    return __generator(this, function (_a) {
      switch (_a.label) {
        case 0:
          _a.trys.push([0, 3, , 4]);
          console.log(''.concat(const_1.BASE_URL, '/data/').concat(choice));
          return [4 /*yield*/, fetch(''.concat(const_1.BASE_URL, '/data/').concat(choice))];
        case 1:
          response = _a.sent();
          if (!response.ok) {
            return [
              2 /*return*/,
              {
                error: {
                  message: 'Failed to fetch data: '.concat(response.statusText),
                  status: response.status,
                },
              },
            ];
          }
          return [4 /*yield*/, response.json()];
        case 2:
          data = _a.sent();
          return [2 /*return*/, { data: data }];
        case 3:
          err_1 = _a.sent();
          return [
            2 /*return*/,
            {
              error: {
                message: err_1 instanceof Error ? err_1.message : 'An unexpected error occurred',
              },
            },
          ];
        case 4:
          return [2 /*return*/];
      }
    });
  });
}
function Dummy() {
  var _a = (0, react_1.useState)(),
    exampleData = _a[0],
    setExampleData = _a[1];
  var _b = (0, react_1.useState)(),
    dataChoice = _b[0],
    setDataChoice = _b[1];
  var _c = (0, react_1.useState)(),
    error = _c[0],
    setError = _c[1];
  var _d = (0, react_1.useState)(false),
    isLoading = _d[0],
    setIsLoading = _d[1];
  (0, react_1.useEffect)(
    function () {
      function fetchData() {
        return __awaiter(this, void 0, void 0, function () {
          var result;
          return __generator(this, function (_a) {
            switch (_a.label) {
              case 0:
                if (!dataChoice) return [2 /*return*/];
                setIsLoading(true);
                setError(undefined);
                return [4 /*yield*/, postPoints(dataChoice)];
              case 1:
                result = _a.sent();
                if (result.error) {
                  setError(result.error.message);
                  setExampleData(undefined);
                } else if (result.data) {
                  setExampleData(result.data);
                }
                setIsLoading(false);
                return [2 /*return*/];
            }
          });
        });
      }
      fetchData();
    },
    [dataChoice],
  );
  function choiceMade(choice) {
    setDataChoice(choice);
  }
  return (
    <div className="flex h-screen w-screen flex-col items-center bg-[#2692db] font-['Neucha']">
      <header className="z-10 mt-24 mb-8 text-center text-4xl text-white uppercase lg:my-4 lg:text-3xl">
        K-Means clustering
      </header>
      <DataChoice_1.default onChoiceMade={choiceMade} />

      {isLoading && <div className="mt-4 text-lg text-white">Loading data...</div>}

      {error && <div className="mt-4 rounded-lg bg-red-500 p-4 text-white shadow-lg">{error}</div>}

      {!isLoading && !error && (
        <ScatterPlot_1.default width={1100} height={550} data={exampleData} />
      )}
    </div>
  );
}
